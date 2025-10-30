# Multi-Task Training Script für "Häuser"-Datensatz
# - Multi-label (6 binäre Merkmale) mit BCEWithLogitsLoss
# - Multi-class (Verglasungstyp: 3 Klassen) mit CrossEntropyLoss (ignore_index für fehlende Werte)
# Hinweis: in Colab vorher Google Drive mounten (siehe unten)


# Vorbereitung Colab: 
!pip install -q gdown

# Google Drive Datei-ID (ZIP)
file_id = "1EuX-9G_Rpae7n7YgcztBjfxOUbofpGKQ"
output_zip = "dataset.zip"

# Download Datensatz für DLM
!gdown --id {file_id} -O {output_zip}

# Entpacke Datensatz
!unzip -q {output_zip} -d data

# Setting up Environment
import os
import glob
import random
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import math
from sklearn.model_selection import train_test_split
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T


# optional: sklearn für Metriken
try:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
except Exception:
    !pip install -q scikit-learn
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error


# Settings & Mapping
ROOT_DIR = "data/Test_Training_DATA_all_images_small"
RESULTS_DIR="DLM_predictions"
CHARTS_DIR="Performance Metrics"
print("ROOT_DIR gesetzt auf:", ROOT_DIR)
BATCH_SIZE = 16
IMG_SIZE = 224   # Empfohlene Größe (kann kleiner sein, z.B. 128, aber 224 ist üblich)
NUM_EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Spaltennamen aus CSV definieren: 
# Titel
TITLE_COLUMN = "Adresse"

# Heizbedarf
HEATING_DEMAND = "Jahresbedarf [kWh]"

# Label-Spalten binärer Klassen (wie in CSV)
BINARY_LABEL_COLUMNS = [
    "Aufsparrendämmung?",
    "Dach gedämmt?",
    "Dach saniert?",
    "Fassadendämmung",
    "Sockeldämmung",
    "Fenster fassadenbündig"
]

# Label-Spalten für multi-class-variable "Verglasungstyp"
GLASS_COL = "Verglasungstyp"  # Werte: "Einfachverglasung","Zweifachverglasung","Dreifachverglasung"
GLASS_MAP = {"Einfachverglasung": 0, "Zweifachverglasung": 1, "Dreifachverglasung": 2}
IGNORE_GLASS_LABEL = -100  # wird bei fehlender Angabe verwendet und an CrossEntropyLoss als ignore_index übergeben


# Da Heizwärmebedarf mit Werten arbeitet, die um einige Dimensionen größer sind und die Losses somit stärker beeinflussen als die anderen Labels (1/0): Einstellung eines Gewichts für diesen:
HEATING_LOSS_WEIGHT = 0.0001 # kann angepasst werden - kleiner Wert, da Heizbedarf numerisch sehr groß ist (z.B. 5000) und sonst den Loss dominiert

# Dataset
class HousesDataset(Dataset):
    def __init__(self, root_dir, label_cols=BINARY_LABEL_COLUMNS, glass_col=GLASS_COL, transform=None,
                 image_extensions=('.jpg', '.jpeg', '.png')):
        self.root = root_dir
        # Unterordner, jeweils ein Data-Punkt (Haus)
        self.folders = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.label_cols = label_cols
        self.glass_col = glass_col
        self.image_extensions = [ext.lower() for ext in image_extensions]  # Unterstützte Bilderweiterungen
        self.transform = transform or T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            # optional: Norm mit ImageNet-Werten (empfohlen bei Pretrained Backbones)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Alle Bilder in den Ordnern finden und indexieren
        self.image_paths = []
        for folder in self.folders:
            folder_images = self._find_all_images(folder)
            if not folder_images:  # Falls keine Bilder gefunden wurden
                print(f"Warnung: Keine Bilder in {folder} gefunden!")
            self.image_paths.append(folder_images)

    def _find_all_images(self, folder):
        """Findet alle Bilder in einem Ordner basierend auf den unterstützten Dateiendungen"""
        images = []
        for f in os.listdir(folder):
            if any(f.lower().endswith(ext) for ext in self.image_extensions):
                images.append(os.path.join(folder, f))
        return images

    def _find_file(self, folder, ext):
        # suche erste Datei mit bestimmter Endung (case-insensitiv)
        for f in os.listdir(folder):
            if f.lower().endswith(ext.lower()):
                return os.path.join(folder, f)
        return None

    def __len__(self):
        # Summe aller Bilder über alle Ordner
        return sum(len(images) for images in self.image_paths)

    def __getitem__(self, idx):
        # Wir müssen den globalen Index auf Ordner und lokales Bild mappen
        folder_idx = 0
        local_idx = idx

        # Finde den richtigen Ordner und das lokale Bildindex
        while local_idx >= len(self.image_paths[folder_idx]):
            local_idx -= len(self.image_paths[folder_idx])
            folder_idx += 1
            if folder_idx >= len(self.folders):
                raise IndexError(f"Index {idx} außerhalb des gültigen Bereichs")

        folder = self.folders[folder_idx]
        img_path = self.image_paths[folder_idx][local_idx]

        # Bild laden
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Fehler beim Laden von {img_path}: {e}")
            # Fallback: Leeres schwarzes Bild
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        # CSV (address_data_csv oder *.csv)
        csv_path = self._find_file(folder, ".csv")
        if csv_path is None:
            # falls keine CSV existiert, erstellen wir leere / all zeros
            df = pd.DataFrame()
        else:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except Exception:
                try:
                    df = pd.read_csv(csv_path)  # fallback
                except Exception:
                    df = pd.DataFrame()

        # Binary labels: 'checked' in der jeweiligen Spalte -> 1, sonst 0
        binary = []
        for col in self.label_cols:
            if col in df.columns and df[col].astype(str).str.contains("checked", case=False, na=False).any():
                binary.append(1.0)
            else:
                binary.append(0.0)
        binary = torch.tensor(binary, dtype=torch.float32)

        # Verglasungstyp -> Map to 0/1/2, otherwise IGNORE_GLASS_LABEL
        glass_label = IGNORE_GLASS_LABEL
        if self.glass_col in df.columns:
            # falls mehrere Zeilen: suche ersten nicht-nan Eintrag
            vals = df[self.glass_col].astype(str).values
            # suche erstes val != nan/empty
            found = None
            for v in vals:
                v_str = str(v).strip()
                if v_str != "" and v_str.lower() != "nan":
                    found = v_str
                    break
            if found is not None:
                if found in GLASS_MAP:
                    glass_label = GLASS_MAP[found]
                else:
                    # evtl. unterschiedliche Schreibweise -> versuche einfache Normalisierung
                    f2 = found.strip()
                    if f2 in GLASS_MAP:
                        glass_label = GLASS_MAP[f2]
                    else:
                        # unbekannter Wert -> IGNORE
                        glass_label = IGNORE_GLASS_LABEL
        glass_label = torch.tensor(glass_label, dtype=torch.long)

        # HEATING_DEMAND (numerisch). Falls vorhanden: erster gültiger numerischer Eintrag, sonst NaN
        heating_value = float("nan")
        if HEATING_DEMAND in df.columns:
            # versuche numerisch zu parsen; Fehler -> NaN
            vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
            found = None
            for v in vals:
                if not (pd.isna(v)):
                    found = v
                    break
            if found is not None:
                heating_value = float(found)

        heating_value = torch.tensor(heating_value, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        # Rückgabe: img, binary, glass_label, heating_value
        return img, binary, glass_label, heating_value
    

# Model (Backbone + drei Heads)
class HouseNet(nn.Module):
    def __init__(self, n_binary_labels=len(BINARY_LABEL_COLUMNS), n_glass_types=len(GLASS_MAP)):
        super().__init__()
        # Erstelle Layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.adapt = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(128*4*4, 256)

        # heads
        self.binary_head = nn.Linear(256, n_binary_labels)
        self.glass_head  = nn.Linear(256, n_glass_types)
        self.heating_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = self.relu(self.conv3(x)); x = self.pool(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))

        b_logits = self.binary_head(x)
        g_logits = self.glass_head(x)
        h_pred   = self.heating_head(x).squeeze(1)

        return b_logits, g_logits, h_pred


# Early Stopping für optimiertes Training
# Early Stopping für optimiertes Training
class EarlyStopping:
    """
    Early Stopping Klasse zur Überwachung der Validation Loss und
    Beendigung des Trainings, wenn keine Verbesserung mehr auftritt.

    Args:
        patience: Anzahl der Epochen, die abgewartet werden, bevor das Training stoppt
        min_delta: Mindestverbesserung, die als signifikant betrachtet wird
        restore_best_weights: Ob die besten Gewichte nach dem Training geladen werden sollen
    """
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        """
        Überprüft, ob das Training gestoppt werden soll.

        Args:
            val_loss: Aktueller Wert der Validation Loss
            model: PyTorch-Modell, dessen Gewichte gespeichert werden sollen

        Returns:
            True, wenn das Training gestoppt werden soll, sonst False
        """
        if self.best_val_loss - val_loss > self.min_delta:
            # Verbesserung gefunden, setze Counter zurück
            self.best_val_loss = val_loss
            self.counter = 0

            # Speichere beste Gewichte
            if self.restore_best_weights:
                self.best_weights = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        else:
            # Keine Verbesserung, erhöhe Counter
            self.counter += 1
            print(f"EarlyStopping: Keine Verbesserung seit {self.counter} Epochen.")

            if self.counter >= self.patience:
                print(f"EarlyStopping: Training wird nach {self.patience} Epochen ohne Verbesserung beendet.")
                self.early_stop = True

        return self.early_stop

    def restore_weights(self, model):
        """
        Stellt die besten Gewichte im Modell wieder her.

        Args:
            model: PyTorch-Modell, dessen Gewichte wiederhergestellt werden sollen
        """
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.best_weights.items()})
            print("EarlyStopping: Beste Gewichte wiederhergestellt.")


# Show Dataset instances
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def evaluate_dataset(root_dir=ROOT_DIR):
    """
    Evaluates the entire dataset and provides statistics about the labels.

    Args:
        root_dir: The root directory containing all the house folders

    Returns:
        A dictionary with dataset statistics
    """
    # Find all house folders
    all_folders = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d))])

    print(f"Analyzing {len(all_folders)} houses in {root_dir}...")

    # Initialize counters
    binary_counts = {col: {"checked": 0, "unchecked": 0, "missing": 0} for col in BINARY_LABEL_COLUMNS}
    glass_counts = {glass_type: 0 for glass_type in GLASS_MAP.keys()}
    glass_counts["MISSING"] = 0

    heating_stats = {
        "available": 0,
        "missing": 0,
        "min": float('inf'),
        "max": float('-inf'),
        "mean": 0,
        "median": None,
        "std": None,
        "values": []
    }

    # Process each folder
    for folder in tqdm(all_folders, desc="Processing houses"):
        # Find CSV file
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break

        if csv_path is None:
            # No CSV - all values count as missing
            for col in BINARY_LABEL_COLUMNS:
                binary_counts[col]["missing"] += 1
            glass_counts["MISSING"] += 1
            heating_stats["missing"] += 1
            continue

        # Load CSV
        try:
            df = pd.read_csv(csv_path, sep=";")
        except Exception:
            try:
                df = pd.read_csv(csv_path)  # fallback
            except Exception:
                # Invalid CSV - all values count as missing
                for col in BINARY_LABEL_COLUMNS:
                    binary_counts[col]["missing"] += 1
                glass_counts["MISSING"] += 1
                heating_stats["missing"] += 1
                continue

        # Process binary columns
        for col in BINARY_LABEL_COLUMNS:
            if col in df.columns:
                if df[col].astype(str).str.contains("checked", case=False, na=False).any():
                    binary_counts[col]["checked"] += 1
                else:
                    binary_counts[col]["unchecked"] += 1
            else:
                binary_counts[col]["missing"] += 1

        # Process glass type
        glass_found = False
        if GLASS_COL in df.columns:
            # Find first non-empty value
            vals = df[GLASS_COL].astype(str).values
            for v in vals:
                v_str = str(v).strip()
                if v_str != "" and v_str.lower() != "nan":
                    # Try to match with known glass types
                    for glass_type in GLASS_MAP.keys():
                        if v_str == glass_type or v_str.strip() == glass_type:
                            glass_counts[glass_type] += 1
                            glass_found = True
                            break
                    if glass_found:
                        break

        if not glass_found:
            glass_counts["MISSING"] += 1

        # Process heating demand
        heating_found = False
        if HEATING_DEMAND in df.columns:
            vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
            for v in vals:
                if not pd.isna(v):
                    heating_stats["available"] += 1
                    heating_stats["values"].append(float(v))
                    heating_stats["min"] = min(heating_stats["min"], float(v))
                    heating_stats["max"] = max(heating_stats["max"], float(v))
                    heating_found = True
                    break

        if not heating_found:
            heating_stats["missing"] += 1

    # Calculate heating statistics if values are available
    if heating_stats["values"]:
        heating_stats["mean"] = np.mean(heating_stats["values"])
        heating_stats["median"] = np.median(heating_stats["values"])
        heating_stats["std"] = np.std(heating_stats["values"])
    else:
        heating_stats["min"] = None
        heating_stats["max"] = None

    # Print results in tabular format
    print("\n===== DATASET EVALUATION =====")
    print(f"Total houses: {len(all_folders)}")

    # Print binary label statistics
    print("\n1. BINARY LABELS:")
    headers = ["Label", "Checked", "Unchecked", "Missing", "Checked %"]

    # Fix: Split the format strings to avoid using float format for string header
    header_format = "{:<30} {:<10} {:<10} {:<10} {:<10}"
    row_format = "{:<30} {:<10} {:<10} {:<10} {:<10.1f}"

    print(header_format.format(*headers))
    print("-" * 75)

    for col in BINARY_LABEL_COLUMNS:
        total = sum(binary_counts[col].values())
        checked_percent = (binary_counts[col]["checked"] / total) * 100 if total > 0 else 0
        print(row_format.format(
            col,
            binary_counts[col]["checked"],
            binary_counts[col]["unchecked"],
            binary_counts[col]["missing"],
            checked_percent
        ))

    # Print glass type statistics
    print("\n2. GLASS TYPES:")
    headers = ["Type", "Count", "Percentage"]

    # Fix: Split format strings here too
    header_format = "{:<20} {:<10} {:<10}"
    row_format = "{:<20} {:<10} {:<10.1f}"

    print(header_format.format(*headers))
    print("-" * 45)

    total_glass = sum(glass_counts.values())
    for glass_type, count in glass_counts.items():
        percent = (count / total_glass) * 100 if total_glass > 0 else 0
        print(row_format.format(glass_type, count, percent))

    # Print heating demand statistics
    print("\n3. HEATING DEMAND:")
    print(f"Available: {heating_stats['available']} ({heating_stats['available']/len(all_folders)*100:.1f}%)")
    print(f"Missing: {heating_stats['missing']} ({heating_stats['missing']/len(all_folders)*100:.1f}%)")

    if heating_stats["values"]:
        print(f"Range: {heating_stats['min']:.1f} - {heating_stats['max']:.1f} kWh")
        print(f"Mean: {heating_stats['mean']:.1f} kWh")
        print(f"Median: {heating_stats['median']:.1f} kWh")
        print(f"Std Dev: {heating_stats['std']:.1f} kWh")

        # Create histogram of heating values
        plt.figure(figsize=(10, 6))
        plt.hist(heating_stats["values"], bins=20, color='purple', alpha=0.7)
        plt.xlabel('Heating Demand (kWh)')
        plt.ylabel('Count')
        plt.title('Distribution of Heating Demand')
        plt.grid(alpha=0.3)

        # Save histogram if CHARTS_DIR exists
        if 'CHARTS_DIR' in globals():
            ensure_dir(CHARTS_DIR)
            plt.savefig(os.path.join(CHARTS_DIR, "heating_demand_distribution.png"), dpi=300, bbox_inches='tight')

        plt.show()

    return {
        "total_houses": len(all_folders),
        "binary_counts": binary_counts,
        "glass_counts": glass_counts,
        "heating_stats": heating_stats
    }



dataset_stats = evaluate_dataset(ROOT_DIR)

# Erstelle Datasets und DataLoader
def multi_label_stratified_split(folders, train_size=0.7, val_size=0.15, test_size=0.15, random_state=SEED):
    """
    Stratifizierte Aufteilung von Ordnern unter besonderer Berücksichtigung
    der seltensten Labels: Einfachverglasung, Dreifachverglasung und Fenster fassadenbündig.
    Stellt sicher, dass Einfachverglasung mindestens einmal in Train und Val vorkommt.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split-Anteile müssen sich zu 1.0 summieren"

    rare_glass_types = ["Einfachverglasung", "Dreifachverglasung"]
    rare_binary_columns = ["Fenster fassadenbündig"]

    # Sammle Labels für alle Ordner
    folder_labels = {}
    einfach_folders = []
    for folder in folders:
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break
        glass_type = "MISSING"
        has_rare_binary = False
        if csv_path is not None:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except:
                try:
                    df = pd.read_csv(csv_path)
                except:
                    df = pd.DataFrame()
            if GLASS_COL in df.columns:
                vals = df[GLASS_COL].astype(str).values
                for v in vals:
                    v_str = str(v).strip()
                    if v_str != "" and v_str.lower() != "nan":
                        if v_str in GLASS_MAP:
                            glass_type = v_str
                            break
                        elif v_str.strip() in GLASS_MAP:
                            glass_type = v_str.strip()
                            break
            for col in rare_binary_columns:
                if col in df.columns and df[col].astype(str).str.contains("checked", case=False, na=False).any():
                    has_rare_binary = True
                    break
        folder_labels[folder] = {
            "glass": glass_type,
            "has_rare_binary": has_rare_binary
        }
        if glass_type == "Einfachverglasung":
            einfach_folders.append(folder)

    # Stratifizierungslabels
    strat_labels = []
    folders_list = list(folder_labels.keys())
    for folder in folders_list:
        labels = folder_labels[folder]
        if labels["glass"] == "Einfachverglasung":
            strat_group = "A_einfach"
        elif labels["glass"] == "Dreifachverglasung":
            strat_group = "B_dreifach"
        elif labels["has_rare_binary"]:
            strat_group = "C_rare_binary"
        elif labels["glass"] != "MISSING":
            strat_group = f"D_{labels['glass']}"
        else:
            strat_group = "E_other"
        strat_labels.append(strat_group)

    # Split
    train_val_size = train_size + val_size
    test_size_adjusted = test_size / 1.0
    val_size_adjusted = val_size / train_val_size

    train_val_folders, test_list, _, _ = train_test_split(
        folders_list, strat_labels,
        test_size=test_size_adjusted,
        random_state=random_state,
        stratify=strat_labels
    )
    train_val_labels = [strat_labels[folders_list.index(folder)] for folder in train_val_folders]
    train_list, val_list, _, _ = train_test_split(
        train_val_folders,
        train_val_labels,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_labels
    )

    # Sicherstellen, dass Einfachverglasung in Train und Val ist
    def has_einfach(split):
        return any(folder_labels[f]["glass"] == "Einfachverglasung" for f in split)

    # Falls nicht vorhanden, verschiebe aus Test/Val/Train
    if not has_einfach(train_list):
        # Suche einen Ordner mit Einfachverglasung in val_list oder test_list
        candidates = [f for f in val_list + test_list if folder_labels[f]["glass"] == "Einfachverglasung"]
        if candidates:
            f_move = candidates[0]
            train_list.append(f_move)
            if f_move in val_list:
                val_list.remove(f_move)
            if f_move in test_list:
                test_list.remove(f_move)
    if not has_einfach(val_list):
        candidates = [f for f in train_list + test_list if folder_labels[f]["glass"] == "Einfachverglasung"]
        if candidates:
            f_move = candidates[0]
            val_list.append(f_move)
            if f_move in train_list:
                train_list.remove(f_move)
            if f_move in test_list:
                test_list.remove(f_move)

    # Ausgabe der finalen Aufteilung
    print(f"\nFinale Aufteilung (nach Sicherstellung von Einfachverglasung):")
    print(f"  Train: {len(train_list)} Ordner")
    print(f"  Val:   {len(val_list)} Ordner")
    print(f"  Test:  {len(test_list)} Ordner")
    print(f"  Einfachverglasung in Train: {has_einfach(train_list)}")
    print(f"  Einfachverglasung in Val:   {has_einfach(val_list)}")

    return train_list, val_list, test_list

# Stratifizierte Aufteilung durchführen
all_folders = sorted([os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR)
                     if os.path.isdir(os.path.join(ROOT_DIR, d))])
train_list, val_list, test_list = multi_label_stratified_split(
    all_folders, train_size=0.7, val_size=0.15, test_size=0.15
)

# temporäre kleine Dataset-Wrapper, damit Listen übergeben werden können
class FoldersDataset(HousesDataset):
    def __init__(self, folders_list, *args, **kwargs):
        # Rufe den Konstruktor der Elternklasse auf
        super().__init__(root_dir=ROOT_DIR, *args, **kwargs)

        # Überschreibe die Ordnerliste mit der übergebenen Liste
        self.folders = folders_list

        # Wichtig: image_paths neu berechnen für die neue Ordnerliste
        self.image_paths = []
        for folder in self.folders:
            folder_images = self._find_all_images(folder)
            if not folder_images:  # Falls keine Bilder gefunden wurden
                print(f"Warnung: Keine Bilder in {folder} gefunden!")
            self.image_paths.append(folder_images)

        print(f"FoldersDataset initialisiert mit {len(self.folders)} Ordnern und {self.__len__()} Bildern.")


# Create all datasets
train_ds = FoldersDataset(train_list)
val_ds   = FoldersDataset(val_list)
test_ds  = FoldersDataset(test_list)

# Create all dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Dataset sizes: Train: {len(train_ds)}, Validation: {len(val_ds)}, Test: {len(test_ds)}")

# Darstellung der Labelverteilung in den Trainingsdaten
def evaluate_dataset_splits(train_ds, val_ds, test_ds):
    """
    Evaluates the label distribution in train, validation and test datasets.
    Shows both absolute counts and percentages for all metrics.

    Args:
        train_ds, val_ds, test_ds: The dataset objects

    Returns:
        Dictionary with statistics about label distributions in each split
    """
    datasets = {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "all": None
    }

    # Initialize counters for each split
    stats = {}
    for split_name in datasets.keys():
        stats[split_name] = {
            "count": 0,
            "binary_counts": {col: {"checked": 0, "unchecked": 0} for col in BINARY_LABEL_COLUMNS},
            "glass_counts": {glass_type: 0 for glass_type in GLASS_MAP.keys()},
            "glass_counts_missing": 0,
            "heating_available": 0,
            "heating_missing": 0,
            "heating_values": []
        }

    # Process each dataset
    for split_name, dataset in datasets.items():
        if split_name == "all":
            continue  # We'll compute 'all' at the end

        print(f"Analyzing {split_name} split ({len(dataset)} instances)...")
        stats[split_name]["count"] = len(dataset)

        # Iterate through dataset and count labels
        for i in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
            _, bin_labels, glass_label, heating_value = dataset[i]

            # Binary labels
            for j, col in enumerate(BINARY_LABEL_COLUMNS):
                if bin_labels[j] > 0.5:  # Checked
                    stats[split_name]["binary_counts"][col]["checked"] += 1
                else:  # Unchecked
                    stats[split_name]["binary_counts"][col]["unchecked"] += 1

            # Glass type
            if glass_label.item() != IGNORE_GLASS_LABEL:
                # Find the glass type name
                for glass_type, idx in GLASS_MAP.items():
                    if idx == glass_label.item():
                        stats[split_name]["glass_counts"][glass_type] += 1
                        break
            else:
                stats[split_name]["glass_counts_missing"] += 1

            # Heating demand
            if not torch.isnan(heating_value):
                stats[split_name]["heating_available"] += 1
                stats[split_name]["heating_values"].append(float(heating_value.item()))
            else:
                stats[split_name]["heating_missing"] += 1

    # Compute 'all' by combining stats from all splits
    stats["all"]["count"] = sum(stats[split]["count"] for split in ["train", "val", "test"])

    # Binary labels
    for col in BINARY_LABEL_COLUMNS:
        stats["all"]["binary_counts"][col]["checked"] = sum(
            stats[split]["binary_counts"][col]["checked"] for split in ["train", "val", "test"]
        )
        stats["all"]["binary_counts"][col]["unchecked"] = sum(
            stats[split]["binary_counts"][col]["unchecked"] for split in ["train", "val", "test"]
        )

    # Glass types
    for glass_type in GLASS_MAP.keys():
        stats["all"]["glass_counts"][glass_type] = sum(
            stats[split]["glass_counts"][glass_type] for split in ["train", "val", "test"]
        )
    stats["all"]["glass_counts_missing"] = sum(
        stats[split]["glass_counts_missing"] for split in ["train", "val", "test"]
    )

    # Heating demand
    stats["all"]["heating_available"] = sum(
        stats[split]["heating_available"] for split in ["train", "val", "test"]
    )
    stats["all"]["heating_missing"] = sum(
        stats[split]["heating_missing"] for split in ["train", "val", "test"]
    )
    stats["all"]["heating_values"] = []
    for split in ["train", "val", "test"]:
        stats["all"]["heating_values"].extend(stats[split]["heating_values"])

    # Print results in tabular format
    print("\n===== DATASET SPLIT EVALUATION =====")

    # Print dataset sizes
    print("\nDataset sizes:")
    for split in ["train", "val", "test", "all"]:
        print(f"{split.capitalize():5s}: {stats[split]['count']:5d} instances")

    # Print binary label statistics
    print("\n1. BINARY LABELS:")
    header_format = "{:<30} {:<20} {:<20} {:<20} {:<20}"
    row_format = "{:<30} {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%)"

    print(header_format.format("Label", "Train", "Val", "Test", "All"))
    print("-" * 110)

    for col in BINARY_LABEL_COLUMNS:
        train_count = stats["train"]["binary_counts"][col]["checked"]
        val_count = stats["val"]["binary_counts"][col]["checked"]
        test_count = stats["test"]["binary_counts"][col]["checked"]
        all_count = stats["all"]["binary_counts"][col]["checked"]

        train_pct = (train_count / stats["train"]["count"]) * 100 if stats["train"]["count"] > 0 else 0
        val_pct = (val_count / stats["val"]["count"]) * 100 if stats["val"]["count"] > 0 else 0
        test_pct = (test_count / stats["test"]["count"]) * 100 if stats["test"]["count"] > 0 else 0
        all_pct = (all_count / stats["all"]["count"]) * 100 if stats["all"]["count"] > 0 else 0

        print(row_format.format(
            col,
            train_count, train_pct,
            val_count, val_pct,
            test_count, test_pct,
            all_count, all_pct
        ))

    # Print glass type statistics
    print("\n2. GLASS TYPES:")
    header_format = "{:<20} {:<20} {:<20} {:<20} {:<20}"
    row_format = "{:<20} {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%)"

    print(header_format.format("Type", "Train", "Val", "Test", "All"))
    print("-" * 100)

    for glass_type in sorted(GLASS_MAP.keys()):
        train_count = stats["train"]["glass_counts"][glass_type]
        val_count = stats["val"]["glass_counts"][glass_type]
        test_count = stats["test"]["glass_counts"][glass_type]
        all_count = stats["all"]["glass_counts"][glass_type]

        train_pct = (train_count / stats["train"]["count"]) * 100 if stats["train"]["count"] > 0 else 0
        val_pct = (val_count / stats["val"]["count"]) * 100 if stats["val"]["count"] > 0 else 0
        test_pct = (test_count / stats["test"]["count"]) * 100 if stats["test"]["count"] > 0 else 0
        all_pct = (all_count / stats["all"]["count"]) * 100 if stats["all"]["count"] > 0 else 0

        print(row_format.format(
            glass_type,
            train_count, train_pct,
            val_count, val_pct,
            test_count, test_pct,
            all_count, all_pct
        ))

    # Add missing values row
    train_count = stats["train"]["glass_counts_missing"]
    val_count = stats["val"]["glass_counts_missing"]
    test_count = stats["test"]["glass_counts_missing"]
    all_count = stats["all"]["glass_counts_missing"]

    train_pct = (train_count / stats["train"]["count"]) * 100 if stats["train"]["count"] > 0 else 0
    val_pct = (val_count / stats["val"]["count"]) * 100 if stats["val"]["count"] > 0 else 0
    test_pct = (test_count / stats["test"]["count"]) * 100 if stats["test"]["count"] > 0 else 0
    all_pct = (all_count / stats["all"]["count"]) * 100 if stats["all"]["count"] > 0 else 0

    print(row_format.format(
        "MISSING",
        train_count, train_pct,
        val_count, val_pct,
        test_count, test_pct,
        all_count, all_pct
    ))

    # Print heating demand statistics
    print("\n3. HEATING DEMAND:")
    header_format = "{:<20} {:<20} {:<20} {:<20} {:<20}"
    row_format = "{:<20} {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%) {:<8d} ({:5.1f}%)"

    print(header_format.format("Status", "Train", "Val", "Test", "All"))
    print("-" * 100)

    # Available
    train_count = stats["train"]["heating_available"]
    val_count = stats["val"]["heating_available"]
    test_count = stats["test"]["heating_available"]
    all_count = stats["all"]["heating_available"]

    train_pct = (train_count / stats["train"]["count"]) * 100 if stats["train"]["count"] > 0 else 0
    val_pct = (val_count / stats["val"]["count"]) * 100 if stats["val"]["count"] > 0 else 0
    test_pct = (test_count / stats["test"]["count"]) * 100 if stats["test"]["count"] > 0 else 0
    all_pct = (all_count / stats["all"]["count"]) * 100 if stats["all"]["count"] > 0 else 0

    print(row_format.format(
        "Available",
        train_count, train_pct,
        val_count, val_pct,
        test_count, test_pct,
        all_count, all_pct
    ))

    # Missing
    train_count = stats["train"]["heating_missing"]
    val_count = stats["val"]["heating_missing"]
    test_count = stats["test"]["heating_missing"]
    all_count = stats["all"]["heating_missing"]

    train_pct = (train_count / stats["train"]["count"]) * 100 if stats["train"]["count"] > 0 else 0
    val_pct = (val_count / stats["val"]["count"]) * 100 if stats["val"]["count"] > 0 else 0
    test_pct = (test_count / stats["test"]["count"]) * 100 if stats["test"]["count"] > 0 else 0
    all_pct = (all_count / stats["all"]["count"]) * 100 if stats["all"]["count"] > 0 else 0

    print(row_format.format(
        "Missing",
        train_count, train_pct,
        val_count, val_pct,
        test_count, test_pct,
        all_count, all_pct
    ))

    # Print heating value statistics for each split
    if stats["all"]["heating_values"]:
        print("\nHeating demand statistics:")
        header_format = "{:<10} {:<15} {:<15} {:<15} {:<15}"
        row_format = "{:<10} {:<15.1f} {:<15.1f} {:<15.1f} {:<15.1f}"

        print(header_format.format("Metric", "Train", "Val", "Test", "All"))
        print("-" * 75)

        for metric, func in [
            ("Min", min),
            ("Max", max),
            ("Mean", lambda x: sum(x)/len(x) if x else 0),
            ("Median", lambda x: sorted(x)[len(x)//2] if x else 0)
        ]:
            train_val = func(stats["train"]["heating_values"]) if stats["train"]["heating_values"] else float('nan')
            val_val = func(stats["val"]["heating_values"]) if stats["val"]["heating_values"] else float('nan')
            test_val = func(stats["test"]["heating_values"]) if stats["test"]["heating_values"] else float('nan')
            all_val = func(stats["all"]["heating_values"]) if stats["all"]["heating_values"] else float('nan')

            print(row_format.format(metric, train_val, val_val, test_val, all_val))

        # Create histograms of heating values for each split
        plt.figure(figsize=(16, 5))

        for i, split in enumerate(["train", "val", "test"]):
            if stats[split]["heating_values"]:
                plt.subplot(1, 3, i+1)
                plt.hist(stats[split]["heating_values"], bins=15, alpha=0.7,
                         color=['blue', 'orange', 'green'][i])
                plt.title(f"{split.capitalize()} Split ({len(stats[split]['heating_values'])} samples)")
                plt.xlabel('Heating Demand (kWh)')
                plt.ylabel('Count')
                plt.grid(alpha=0.3)

        plt.tight_layout()

        # Save histogram if CHARTS_DIR exists
        if 'CHARTS_DIR' in globals():
            ensure_dir(CHARTS_DIR)
            plt.savefig(os.path.join(CHARTS_DIR, "heating_demand_by_split.png"), dpi=300, bbox_inches='tight')

        plt.show()

    return stats

# Example usage:
split_stats = evaluate_dataset_splits(train_ds, val_ds, test_ds)

# Darstellung eines Testbildes
def show_random_train_image_with_title(train_folders, title_col=TITLE_COLUMN):
    ds = FoldersDataset(train_folders)
    idx = random.randint(0, len(ds)-1)

    # unpack 4 values (img, binary, glass, heating)
    sample = ds[idx]
    if len(sample) == 4:
        img_item, bin_labels, glass_label, heating_value = sample
    else:
        raise RuntimeError(f"Dataset returned {len(sample)} items; erwartet 4 (img, bin, glass, heating). Bitte Klassen neu laden.")

    # Finde den richtigen Ordner und lokalen Bildindex basierend auf dem globalen Index
    folder_idx = 0
    local_idx = idx
    while local_idx >= len(ds.image_paths[folder_idx]):
        local_idx -= len(ds.image_paths[folder_idx])
        folder_idx += 1

    folder = ds.folders[folder_idx]
    img_path = ds.image_paths[folder_idx][local_idx]

    # Titel extrahieren (falls CSV vorhanden)
    def _find_csv(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                return os.path.join(folder, f)
        return None

    title_val = None
    csv_path = _find_csv(folder)
    if csv_path:
        try:
            df = pd.read_csv(csv_path, sep=";")
            if title_col in df.columns:
                for v in df[title_col].astype(str).values:
                    if str(v).strip() != "" and str(v).lower() != "nan":
                        title_val = str(v).strip()
                        break
        except Exception:
            title_val = None

    if title_val is None:
        title_val = os.path.basename(folder.rstrip("/\\"))

    # --- Bild-Typ prüfen und in numpy-Array für plt umwandeln ---
    # Falls img_item ein PIL.Image -> direkt anzeigen
    if isinstance(img_item, Image.Image):
        img_np = np.array(img_item) / 255.0
    elif isinstance(img_item, torch.Tensor):
        # Tensor: Annahme: [C,H,W], normalisiert mit ImageNet mean/std
        img_np = img_item.cpu().numpy()
        # Falls Transform beinhaltete ToTensor + Normalize
        if img_np.ndim == 3:
            img_np = img_np.transpose(1,2,0)  # H,W,C
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std) + mean
            img_np = np.clip(img_np, 0, 1)
        else:
            raise RuntimeError("Unerwartete Tensor-Form für Bild: shape=" + str(img_np.shape))
    else:
        # falls es ein numpy-array ist
        try:
            img_np = np.array(img_item)
            if img_np.dtype == np.uint8:
                img_np = img_np / 255.0
        except Exception as e:
            raise RuntimeError("Unbekannter Bildtyp: " + str(type(img_item))) from e

    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.axis('off')

    bin_labels_list = {name: int(val) for name, val in zip(BINARY_LABEL_COLUMNS, bin_labels.cpu().numpy())}
    inv_glass_map = {v:k for k,v in GLASS_MAP.items()}
    glass_text = inv_glass_map.get(int(glass_label.cpu().item()), "MISSING") if int(glass_label.cpu().item()) != IGNORE_GLASS_LABEL else "MISSING"

    heating_text = f"Heating: {float(heating_value.cpu().item()):.1f} kWh" if not np.isnan(float(heating_value.cpu().item())) else "Heating: MISSING"

    # Zeige auch Bildpfad und Index-Informationen
    img_filename = os.path.basename(img_path)

    title_str = f"{title_val}\nBild: {img_filename}\n{heating_text}\nGlass: {glass_text}\n" + ", ".join([f"{k}:{v}" for k,v in bin_labels_list.items()])
    plt.title(title_str, fontsize=9)
    plt.show()

# Aufruf:
show_random_train_image_with_title(train_list)


# Model, Loss, Optimizer
model = HouseNet().to(DEVICE)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_glass  = nn.CrossEntropyLoss(ignore_index=IGNORE_GLASS_LABEL)
criterion_heating = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# Training & Validation
def evaluate(model, loader):
    model.eval()
    all_bin_targets = []
    all_bin_preds = []
    all_glass_targets = []
    all_glass_preds = []
    all_heating_targets = []
    all_heating_preds = []
    losses = []

    with torch.no_grad():
        for imgs, bin_labels, glass_labels, heating_labels in loader:
            imgs = imgs.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE)
            glass_labels = glass_labels.to(DEVICE)
            heating_labels = heating_labels.to(DEVICE)

            b_logits, g_logits, h_preds = model(imgs)
            loss_b = criterion_binary(b_logits, bin_labels)

            if (glass_labels != IGNORE_GLASS_LABEL).any():
                loss_g = criterion_glass(g_logits, glass_labels)
            else:
                loss_g = torch.tensor(0.0, device=DEVICE)

            # heating loss only for valid targets
            mask_heating = ~torch.isnan(heating_labels)
            if mask_heating.any():
                # select valid preds/targets
                h_pred_sel = h_preds[mask_heating]
                h_target_sel = heating_labels[mask_heating]
                loss_h = criterion_heating(h_pred_sel, h_target_sel)
            else:
                loss_h = torch.tensor(0.0, device=DEVICE)

            losses.append((loss_b.item(), loss_g.item(), loss_h.item()))

            # preds for binary
            probs = torch.sigmoid(b_logits).cpu().numpy()
            preds_bin = (probs > 0.5).astype(int)
            all_bin_preds.append(preds_bin)
            all_bin_targets.append(bin_labels.cpu().numpy().astype(int))

            # glass preds
            glass_targets_np = glass_labels.cpu().numpy()
            glass_pred_np = np.argmax(g_logits.cpu().numpy(), axis=1)
            all_glass_targets.append(glass_targets_np)
            all_glass_preds.append(glass_pred_np)

            # heating preds/targets (store NaNs as-is)
            all_heating_preds.append(h_preds.cpu().numpy())
            all_heating_targets.append(heating_labels.cpu().numpy())

    # concat
    all_bin_targets = np.vstack(all_bin_targets) if all_bin_targets else np.zeros((0, len(BINARY_LABEL_COLUMNS)), dtype=int)
    all_bin_preds = np.vstack(all_bin_preds) if all_bin_preds else np.zeros_like(all_bin_targets)

    all_glass_targets = np.concatenate(all_glass_targets) if all_glass_targets else np.array([], dtype=int)
    all_glass_preds = np.concatenate(all_glass_preds) if all_glass_preds else np.array([], dtype=int)

    all_heating_targets = np.concatenate(all_heating_targets) if all_heating_targets else np.array([], dtype=float)
    all_heating_preds   = np.concatenate(all_heating_preds)   if all_heating_preds   else np.array([], dtype=float)

    # Binary per-class metrics
    prf_bin = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average=None, zero_division=0)

    # Glass metrics (ignore IGNORE index)
    mask_valid = all_glass_targets != IGNORE_GLASS_LABEL
    if mask_valid.sum() > 0:
        glass_acc = accuracy_score(all_glass_targets[mask_valid], all_glass_preds[mask_valid])
        prf_glass = precision_recall_fscore_support(all_glass_targets[mask_valid], all_glass_preds[mask_valid], average=None, zero_division=0)
    else:
        glass_acc = None
        prf_glass = None

    # Heating metrics: filter out NaNs
    heating_mask = ~np.isnan(all_heating_targets)
    if heating_mask.sum() > 0:
        heating_mae = mean_absolute_error(all_heating_targets[heating_mask], all_heating_preds[heating_mask])
        heating_rmse = math.sqrt(np.mean((all_heating_targets[heating_mask] - all_heating_preds[heating_mask])**2))
    else:
        heating_mae = None
        heating_rmse = None

    avg_loss_b = np.mean([x[0] for x in losses]) if losses else 0.0
    avg_loss_g = np.mean([x[1] for x in losses]) if losses else 0.0
    avg_loss_h = np.mean([x[2] for x in losses]) if losses else 0.0

    return {
        "binary_prf": prf_bin,
        "glass_prf": prf_glass,
        "glass_acc": glass_acc,
        "loss_b": avg_loss_b,
        "loss_g": avg_loss_g,
        "loss_h": avg_loss_h,
        "heating_mae": heating_mae,
        "heating_rmse": heating_rmse
    }




# Training loop mit Early Stopping
train_losses_epochs = []
val_loss_b_epochs = []
val_loss_g_epochs = []
val_loss_h_epochs = []
val_total_loss_epochs = []
val_glass_acc_epochs = []
val_heating_mae_epochs = []

# Early Stopping initialisieren
early_stopping = EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True)

# Initialisiere best_val_loss mit einem sehr hohen Wert
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    running_loss_b = 0.0
    running_loss_g = 0.0
    running_loss_h = 0.0
    iters = 0
    for imgs, bin_labels, glass_labels, heating_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        imgs = imgs.to(DEVICE)
        bin_labels = bin_labels.to(DEVICE)
        glass_labels = glass_labels.to(DEVICE)
        heating_labels = heating_labels.to(DEVICE)

        optimizer.zero_grad()
        b_logits, g_logits, h_preds = model(imgs)

        loss_b = criterion_binary(b_logits, bin_labels)

        if (glass_labels != IGNORE_GLASS_LABEL).any():
            loss_g = criterion_glass(g_logits, glass_labels)
        else:
            loss_g = torch.tensor(0.0, device=DEVICE)

        mask_heating = ~torch.isnan(heating_labels)
        if mask_heating.any():
            h_pred_sel = h_preds[mask_heating]
            h_target_sel = heating_labels[mask_heating]
            loss_h = criterion_heating(h_pred_sel, h_target_sel)
        else:
            loss_h = torch.tensor(0.0, device=DEVICE)

        loss = loss_b + loss_g + loss_h * HEATING_LOSS_WEIGHT
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_b += loss_b.item()
        running_loss_g += loss_g.item()
        running_loss_h += loss_h.item()
        iters += 1

    avg_train_loss = running_loss / max(1, iters)
    avg_train_loss_b = running_loss_b / max(1, iters)
    avg_train_loss_g = running_loss_g / max(1, iters)
    avg_train_loss_h = running_loss_h / max(1, iters)

    train_losses_epochs.append(avg_train_loss)

    # Validation
    val_stats = evaluate(model, val_loader)

    # Metriken für Plotting aufzeichnen
    val_loss_b_epochs.append(val_stats['loss_b'])
    val_loss_g_epochs.append(val_stats['loss_g'])
    val_loss_h_epochs.append(val_stats['loss_h'])
    total_val_loss = val_stats['loss_b'] + val_stats['loss_g'] + val_stats['loss_h']
    val_total_loss_epochs.append(total_val_loss)
    val_glass_acc_epochs.append(val_stats['glass_acc'] if val_stats['glass_acc'] is not None else np.nan)
    val_heating_mae_epochs.append(val_stats['heating_mae'] if val_stats['heating_mae'] is not None else np.nan)

    # Epoch-Zusammenfassung ausgeben
    print(f"\nEpoch {epoch} summary:")
    print(f"Train Loss: {avg_train_loss:.4f} (B:{avg_train_loss_b:.4f}, G:{avg_train_loss_g:.4f}, H:{avg_train_loss_h:.4f})")
    print(f"Val Loss: {total_val_loss:.4f} (B:{val_stats['loss_b']:.4f}, G:{val_stats['loss_g']:.4f}, H:{val_stats['loss_h']:.4f})")

    # Binary per-class metrics
    prec, rec, f1, sup = val_stats['binary_prf']
    print("Binary labels (per class):")
    for i, col in enumerate(BINARY_LABEL_COLUMNS):
        print(f"  {col:25s}  P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1[i]:.3f}  support={sup[i]}")

    # Glass metrics
    if val_stats['glass_prf'] is not None:
      p_g, r_g, f1_g, s_g = val_stats['glass_prf']
      print(f"Glass accuracy: {val_stats['glass_acc']:.3f}")

      # Ermittle die tatsächlich vorhandenen Klassen
      available_classes = len(p_g)
      glass_names = [name for name, idx in sorted(GLASS_MAP.items(), key=lambda x: x[1])][:available_classes]
      for i, class_name in enumerate(glass_names):
          print(f"  {class_name:20s}  P={p_g[i]:.3f}  R={r_g[i]:.3f}  F1={f1_g[i]:.3f} support={s_g[i]}")
    else:
        print("No valid glass labels in validation set (skipping glass metrics)")

    # Heating metrics
    if val_stats['heating_mae'] is not None:
        print(f"Heating MAE: {val_stats['heating_mae']:.2f}")
        if val_stats['heating_rmse'] is not None:
            print(f"Heating RMSE: {val_stats['heating_rmse']:.2f}")
    else:
        print("No valid heating values in validation set (skipping heating metrics)")

    # Modell-Checkpointing (bestes Modell basierend auf Validation-Loss)
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        torch.save(model.state_dict(), "best_housenet.pth")
        print("Saved best model -> best_housenet.pth")

    # Early Stopping prüfen
    if early_stopping(total_val_loss, model):
        print(f"Early stopping triggered after {epoch} epochs")
        break

# Nach dem Training die besten Gewichte wiederherstellen
early_stopping.restore_weights(model)
print("Training completed - best weights restored.")




# Model Test:
def _find_csv_in_folder(folder):
    for f in os.listdir(folder):
        if f.lower().endswith(".csv"):
            return os.path.join(folder, f)
    return None

def read_title_from_folder(folder, title_col=TITLE_COLUMN):
    csv_path = _find_csv_in_folder(folder)
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path, sep=";")
    except Exception:
        try:
            df = pd.read_csv(csv_path)  # fallback
        except Exception:
            return None
    if title_col in df.columns:
        # return first non-empty value
        for v in df[title_col].astype(str).values:
            if str(v).strip() != "" and str(v).lower() != "nan":
                return str(v).strip()
    return None

def test_model_and_write_results(model, test_loader, folders_list=None, results_dir=RESULTS_DIR):
    """
    Evaluates model using the provided test_loader and writes one CSV per object into results_dir.
    Returns aggregated test_stats (same structure as previous test_model).
    """
    ensure_dir(results_dir)

    # Get the folders list if not provided (e.g., from test_ds.folders)
    folders = folders_list or test_loader.dataset.folders

    model.eval()
    all_bin_targets = []
    all_bin_preds = []
    all_glass_targets = []
    all_glass_preds = []
    all_heating_targets = []
    all_heating_preds = []
    losses = []

    folder_ptr = 0
    inv_glass = {v:k for k,v in GLASS_MAP.items()}

    with torch.no_grad():
        for imgs, bin_labels, glass_labels, heating_labels in test_loader:
            batch_size_actual = imgs.size(0)
            batch_folders = folders[folder_ptr: folder_ptr + batch_size_actual]
            folder_ptr += batch_size_actual

            imgs = imgs.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE)
            glass_labels = glass_labels.to(DEVICE)
            heating_labels = heating_labels.to(DEVICE)

            b_logits, g_logits, h_preds = model(imgs)

            # Calculate losses
            loss_b = criterion_binary(b_logits, bin_labels).item()

            if (glass_labels != IGNORE_GLASS_LABEL).any():
                loss_g = criterion_glass(g_logits, glass_labels).item()
            else:
                loss_g = 0.0

            mask_heating = ~torch.isnan(heating_labels)
            if mask_heating.any():
                h_pred_sel = h_preds[mask_heating]
                h_target_sel = heating_labels[mask_heating]
                loss_h = criterion_heating(h_pred_sel, h_target_sel).item()
            else:
                loss_h = 0.0

            losses.append((loss_b, loss_g, loss_h))

            # Predictions & probs
            probs = torch.sigmoid(b_logits).cpu().numpy()
            preds_bin = (probs > 0.5).astype(int)
            all_bin_preds.append(preds_bin)
            all_bin_targets.append(bin_labels.cpu().numpy().astype(int))

            glass_preds_idx = np.argmax(g_logits.cpu().numpy(), axis=1)
            all_glass_preds.append(glass_preds_idx)
            all_glass_targets.append(glass_labels.cpu().numpy())

            # Heating predictions and targets
            all_heating_preds.append(h_preds.cpu().numpy())
            all_heating_targets.append(heating_labels.cpu().numpy())

            # Write results for each element in the batch
            for i, folder in enumerate(batch_folders):
                base_name = os.path.basename(folder.rstrip("/\\"))
                title_val = read_title_from_folder(folder, title_col=TITLE_COLUMN) or base_name

                row = {
                    "folder": base_name,
                    "title": title_val
                }
                # binary probs + preds
                for j, col in enumerate(BINARY_LABEL_COLUMNS):
                    row[f"{col}_prob"] = float(probs[i, j])
                    row[f"{col}_pred"] = int(preds_bin[i, j])

                # glass
                g_pred_idx = int(glass_preds_idx[i])
                g_pred_name = inv_glass.get(g_pred_idx, "UNKNOWN")
                row["glass_pred_idx"] = g_pred_idx
                row["glass_pred_name"] = g_pred_name

                # original glass label (if present)
                true_glass = int(glass_labels.cpu().numpy()[i])
                if true_glass != IGNORE_GLASS_LABEL:
                    row["glass_true_idx"] = true_glass
                    row["glass_true_name"] = inv_glass.get(true_glass, "UNKNOWN")
                else:
                    row["glass_true_idx"] = IGNORE_GLASS_LABEL
                    row["glass_true_name"] = "MISSING"

                # NEW: heating prediction and true value
                row["heating_pred"] = float(h_preds[i].cpu().item())
                heating_true = float(heating_labels[i].cpu().item())
                if not np.isnan(heating_true):
                    row["heating_true"] = heating_true
                else:
                    row["heating_true"] = "MISSING"

                # write CSV (one file per object)
                out_csv_path = os.path.join(results_dir, f"{base_name}_prediction.csv")
                with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)

    # concat arrays for aggregated metrics
    all_bin_targets = np.vstack(all_bin_targets) if all_bin_targets else np.zeros((0, len(BINARY_LABEL_COLUMNS)), dtype=int)
    all_bin_preds   = np.vstack(all_bin_preds)   if all_bin_preds   else np.zeros_like(all_bin_targets)

    all_glass_targets = np.concatenate(all_glass_targets) if all_glass_targets else np.array([], dtype=int)
    all_glass_preds   = np.concatenate(all_glass_preds)   if all_glass_preds   else np.array([], dtype=int)

    all_heating_targets = np.concatenate(all_heating_targets) if all_heating_targets else np.array([], dtype=float)
    all_heating_preds   = np.concatenate(all_heating_preds)   if all_heating_preds   else np.array([], dtype=float)

    # Binary metrics
    prf_bin = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average=None, zero_division=0)

    # Glass metrics (ignore IGNORE index)
    mask_valid = all_glass_targets != IGNORE_GLASS_LABEL
    if mask_valid.sum() > 0:
        glass_acc = accuracy_score(all_glass_targets[mask_valid], all_glass_preds[mask_valid])
        prf_glass = precision_recall_fscore_support(all_glass_targets[mask_valid], all_glass_preds[mask_valid], average=None, zero_division=0)
    else:
        glass_acc = None
        prf_glass = None

    # Heating metrics
    heating_mask = ~np.isnan(all_heating_targets)
    if heating_mask.sum() > 0:
        heating_mae = mean_absolute_error(all_heating_targets[heating_mask], all_heating_preds[heating_mask])
        heating_rmse = math.sqrt(np.mean((all_heating_targets[heating_mask] - all_heating_preds[heating_mask])**2))
    else:
        heating_mae = None
        heating_rmse = None

    avg_loss_b = np.mean([x[0] for x in losses]) if losses else 0.0
    avg_loss_g = np.mean([x[1] for x in losses]) if losses else 0.0
    avg_loss_h = np.mean([x[2] for x in losses]) if losses else 0.0

    return {
        "binary_prf": prf_bin,
        "glass_prf": prf_glass,
        "glass_acc": glass_acc,
        "loss_b": avg_loss_b,
        "loss_g": avg_loss_g,
        "loss_h": avg_loss_h,
        "heating_mae": heating_mae,
        "heating_rmse": heating_rmse,
        "all_bin_targets": all_bin_targets,
        "all_bin_preds": all_bin_preds,
        "all_glass_targets": all_glass_targets,
        "all_glass_preds": all_glass_preds,
        "all_heating_targets": all_heating_targets,
        "all_heating_preds": all_heating_preds
    }

# Aufruf:
test_stats = test_model_and_write_results(model, test_loader, folders_list=test_list, results_dir=RESULTS_DIR)
print("Test Losses B:{:.4f} G:{:.4f} H:{:.4f}".format(
    test_stats['loss_b'], test_stats['loss_g'], test_stats['loss_h']))

if test_stats['heating_mae'] is not None:
    print(f"Heating MAE: {test_stats['heating_mae']:.2f}, RMSE: {test_stats['heating_rmse']:.2f}")

print("Ergebnisse geschrieben nach:", RESULTS_DIR)

# Performance metrics
# Create a directory for saving charts
ensure_dir(CHARTS_DIR)

# Unified plotting function for training metrics
def plot_training_metrics(train_losses_epochs, val_total_loss_epochs,
                         val_loss_b_epochs=None, val_loss_g_epochs=None, val_loss_h_epochs=None,
                         val_glass_acc_epochs=None, val_heating_mae_epochs=None,
                         binary_prf=None, glass_prf=None, test_stats=None, save_fig=True):
    """
    Comprehensive plotting function that creates all training and evaluation charts
    """
    # 1. OVERVIEW PLOT
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle("Training Overview", fontsize=16)

    # 1a. Overall loss plot
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax1.plot(np.arange(1, len(train_losses_epochs)+1), train_losses_epochs, label='Train Total', color='blue', marker='o')
    ax1.plot(np.arange(1, len(val_total_loss_epochs)+1), val_total_loss_epochs, label='Val Total', color='red', marker='o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss per Epoch')
    ax1.legend()
    ax1.grid(alpha=0.2)

    # 1b. Component losses
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    if val_loss_b_epochs is not None:
        ax2.plot(np.arange(1, len(val_loss_b_epochs)+1), val_loss_b_epochs, label='Binary', color='green', marker='o')
    if val_loss_g_epochs is not None:
        ax2.plot(np.arange(1, len(val_loss_g_epochs)+1), val_loss_g_epochs, label='Glass', color='orange', marker='o')
    if val_loss_h_epochs is not None:
        ax2.plot(np.arange(1, len(val_loss_h_epochs)+1), val_loss_h_epochs, label='Heating', color='purple', marker='o')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Component Losses')
    ax2.legend()
    ax2.grid(alpha=0.2)

    # 1c. Glass Accuracy
    ax3 = plt.subplot2grid((2, 3), (0, 2))
    if val_glass_acc_epochs is not None:
        # Filter out NaN values for plotting
        epochs = np.arange(1, len(val_glass_acc_epochs)+1)
        valid_mask = ~np.isnan(val_glass_acc_epochs)

        if np.any(valid_mask):
            ax3.plot(epochs[valid_mask], np.array(val_glass_acc_epochs)[valid_mask],
                    color='orange', marker='o')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Glass Classification Accuracy')
            ax3.grid(alpha=0.2)
            # Try to set y-axis limits to reasonable values for accuracy
            ax3.set_ylim([0, 1.05])

    # 1d. Heating MAE
    ax4 = plt.subplot2grid((2, 3), (1, 0))
    if val_heating_mae_epochs is not None:
        # Filter out NaN values for plotting
        epochs = np.arange(1, len(val_heating_mae_epochs)+1)
        valid_mask = ~np.isnan(val_heating_mae_epochs)

        if np.any(valid_mask):
            ax4.plot(epochs[valid_mask], np.array(val_heating_mae_epochs)[valid_mask],
                    color='purple', marker='o')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('MAE (kWh)')
            ax4.set_title('Heating Mean Absolute Error')
            ax4.grid(alpha=0.2)

    # 1e. Loss Ratio (Binary:Glass:Heating)
    ax5 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    if all(x is not None for x in [val_loss_b_epochs, val_loss_g_epochs, val_loss_h_epochs]):
        total = np.array(val_loss_b_epochs) + np.array(val_loss_g_epochs) + np.array(val_loss_h_epochs)
        b_ratio = np.array(val_loss_b_epochs) / np.maximum(total, 1e-10)
        g_ratio = np.array(val_loss_g_epochs) / np.maximum(total, 1e-10)
        h_ratio = np.array(val_loss_h_epochs) / np.maximum(total, 1e-10)

        epochs = np.arange(1, len(val_loss_b_epochs)+1)
        ax5.stackplot(epochs, b_ratio, g_ratio, h_ratio,
                     labels=['Binary', 'Glass', 'Heating'],
                     colors=['green', 'orange', 'purple'])
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss Proportion')
        ax5.set_title('Relative Contribution of Each Task to Total Loss')
        ax5.legend()
        ax5.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    # Save overview figure if requested
    if save_fig:
        fig_path = os.path.join(CHARTS_DIR, "training_overview.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved training overview chart to {fig_path}")

    plt.show()

    # 2. BINARY CLASSIFICATION METRICS
    fig1 = plt.figure(figsize=(14, 6))
    plt.suptitle("Binary Classification Metrics", fontsize=16)

    plt.subplot(1, 2, 1)
    if val_loss_b_epochs is not None:
        plt.plot(np.arange(1, len(val_loss_b_epochs)+1), val_loss_b_epochs,
                label='Binary Loss', color='green', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Binary Classification Loss')
        plt.grid(alpha=0.2)

    # Binary metrics per class if available
    if binary_prf is not None:
        plt.subplot(1, 2, 2)

        # Check if binary_prf has the correct shape
        if len(binary_prf) == 4:  # Expected format: (precision, recall, f1, support)
            prec, rec, f1, support = binary_prf

            # Check if the metrics match the number of binary labels
            if len(prec) == len(BINARY_LABEL_COLUMNS):
                x = np.arange(len(BINARY_LABEL_COLUMNS))
                width = 0.25

                plt.bar(x - width, prec, width, label='Precision')
                plt.bar(x, rec, width, label='Recall')
                plt.bar(x + width, f1, width, label='F1')

                plt.xlabel('Binary Classes')
                plt.ylabel('Score')
                plt.title('Binary Classification Metrics per Class')
                plt.xticks(x, [label[:10] + '...' if len(label) > 10 else label for label in BINARY_LABEL_COLUMNS], rotation=45, ha='right')
                plt.legend()
                plt.grid(alpha=0.2)
            else:
                plt.text(0.5, 0.5, f"Error: Metrics dimension ({len(prec)}) doesn't match label count ({len(BINARY_LABEL_COLUMNS)})",
                         ha='center', va='center', transform=plt.gca().transAxes)
                print(f"Warning: Binary metrics dimension ({len(prec)}) doesn't match label count ({len(BINARY_LABEL_COLUMNS)})")
        else:
            plt.text(0.5, 0.5, "Error: Invalid binary metrics format",
                     ha='center', va='center', transform=plt.gca().transAxes)
            print(f"Warning: Invalid binary_prf format. Expected tuple of 4 arrays, got {type(binary_prf)} of length {len(binary_prf)}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    # Save binary plot
    if save_fig:
        fig_path = os.path.join(CHARTS_DIR, "binary_metrics.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved binary metrics chart to {fig_path}")

    plt.show()

    # 3. GLASS CLASSIFICATION METRICS
    fig2 = plt.figure(figsize=(14, 6))
    plt.suptitle("Glass Classification Metrics", fontsize=16)

    plt.subplot(1, 2, 1)
    if val_loss_g_epochs is not None:
        plt.plot(np.arange(1, len(val_loss_g_epochs)+1), val_loss_g_epochs,
                label='Glass Loss', color='orange', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Glass Classification Loss')
        plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)
    if val_glass_acc_epochs is not None:
        # Filter out NaN values for plotting
        epochs = np.arange(1, len(val_glass_acc_epochs)+1)
        valid_mask = ~np.isnan(val_glass_acc_epochs)

        if np.any(valid_mask):
            plt.plot(epochs[valid_mask], np.array(val_glass_acc_epochs)[valid_mask],
                    color='orange', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Glass Classification Accuracy')
            plt.ylim([0, 1.05])
            plt.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    # Save glass plot
    if save_fig:
        fig_path = os.path.join(CHARTS_DIR, "glass_metrics.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved glass metrics chart to {fig_path}")

    plt.show()

    # Glass metrics per class if available
    if glass_prf is not None:
        if len(glass_prf) == 4:  # Expected format: (precision, recall, f1, support)
            prec, rec, f1, _ = glass_prf

            # Get the actual available glass classes (not all 3 might be in the test set)
            available_classes = len(prec)

            if available_classes > 0:
                fig2b = plt.figure(figsize=(10, 6))
                plt.title('Glass Classification Metrics per Class', fontsize=16)

                # Get class names for the available indices
                glass_labels = []
                for i in range(available_classes):
                    for k, v in GLASS_MAP.items():
                        if v == i:
                            glass_labels.append(k)
                            break
                    else:
                        glass_labels.append(f"Class {i}")  # Fallback if class name not found

                x = np.arange(available_classes)
                width = 0.25

                plt.bar(x - width, prec, width, label='Precision')
                plt.bar(x, rec, width, label='Recall')
                plt.bar(x + width, f1, width, label='F1')

                plt.xlabel('Glass Types')
                plt.ylabel('Score')
                plt.xticks(x, glass_labels)
                plt.legend()
                plt.grid(alpha=0.2)
                plt.tight_layout()

                # Save glass class metrics
                if save_fig:
                    fig_path = os.path.join(CHARTS_DIR, "glass_class_metrics.png")
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Saved glass class metrics chart to {fig_path}")

                plt.show()
            else:
                print("Warning: No glass class metrics available to plot")
        else:
            print(f"Warning: Invalid glass_prf format. Expected tuple of 4 arrays, got {type(glass_prf)} of length {len(glass_prf)}")

    # 4. HEATING REGRESSION METRICS
    fig3 = plt.figure(figsize=(14, 6))
    plt.suptitle("Heating Demand Regression Metrics", fontsize=16)

    plt.subplot(1, 2, 1)
    if val_loss_h_epochs is not None:
        plt.plot(np.arange(1, len(val_loss_h_epochs)+1), val_loss_h_epochs,
                label='Heating Loss', color='purple', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Heating Regression Loss')
        plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)
    if val_heating_mae_epochs is not None:
        # Filter out NaN values for plotting
        epochs = np.arange(1, len(val_heating_mae_epochs)+1)
        valid_mask = ~np.isnan(val_heating_mae_epochs)

        if np.any(valid_mask):
            plt.plot(epochs[valid_mask], np.array(val_heating_mae_epochs)[valid_mask],
                    color='purple', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('MAE (kWh)')
            plt.title('Heating Mean Absolute Error')
            plt.grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    # Save heating plot
    if save_fig:
        fig_path = os.path.join(CHARTS_DIR, "heating_metrics.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved heating metrics chart to {fig_path}")

    plt.show()

    # 5. CONFUSION MATRICES
    if test_stats is not None:
        # Binary confusion matrices (one per class)
        for i, label_name in enumerate(BINARY_LABEL_COLUMNS):
            fig = plt.figure(figsize=(6, 5))
            y_true = test_stats['all_bin_targets'][:, i]
            y_pred = test_stats['all_bin_preds'][:, i]

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
            disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
            plt.title(f'Confusion Matrix: {label_name}')
            plt.tight_layout()

            if save_fig:
                # Create a filename-safe version of the label name
                safe_name = "".join(c if c.isalnum() else "_" for c in label_name)
                fig_path = os.path.join(CHARTS_DIR, f"cm_binary_{safe_name}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Saved confusion matrix for {label_name} to {fig_path}")

            plt.show()

        # Glass confusion matrix (if available)
        if 'all_glass_targets' in test_stats and 'all_glass_preds' in test_stats:
            mask_valid = test_stats['all_glass_targets'] != IGNORE_GLASS_LABEL
            if mask_valid.sum() > 0:
                fig = plt.figure(figsize=(8, 6))
                y_true = test_stats['all_glass_targets'][mask_valid]
                y_pred = test_stats['all_glass_preds'][mask_valid]

                # Find unique class values actually present in the test set
                unique_classes = np.unique(np.concatenate((y_true, y_pred)))

                # Get class names for the available indices
                glass_labels = []
                for i in sorted(unique_classes):
                    for k, v in GLASS_MAP.items():
                        if v == i:
                            glass_labels.append(k)
                            break
                    else:
                        glass_labels.append(f"Class {i}")  # Fallback if class name not found

                cm = confusion_matrix(y_true, y_pred, labels=sorted(unique_classes))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=glass_labels)
                disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
                plt.title('Confusion Matrix: Glass Types')
                plt.tight_layout()

                if save_fig:
                    fig_path = os.path.join(CHARTS_DIR, "cm_glass.png")
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Saved glass confusion matrix to {fig_path}")

                plt.show()


# After training and testing, create all visualizations with one call
plot_training_metrics(
    train_losses_epochs=train_losses_epochs,
    val_total_loss_epochs=val_total_loss_epochs,
    val_loss_b_epochs=val_loss_b_epochs,
    val_loss_g_epochs=val_loss_g_epochs,
    val_loss_h_epochs=val_loss_h_epochs,
    val_glass_acc_epochs=val_glass_acc_epochs,
    val_heating_mae_epochs=val_heating_mae_epochs,
    binary_prf=test_stats['binary_prf'],
    glass_prf=test_stats['glass_prf'],
    test_stats=test_stats,
    save_fig=True
)

# Also save confusion matrices for binary and glass predictions
def plot_and_save_confusion_matrices(test_stats, save_fig=True):
    """Plot and optionally save confusion matrices for binary and glass predictions"""
    # Binary confusion matrices (one per class)
    for i, label_name in enumerate(BINARY_LABEL_COLUMNS):
        fig = plt.figure(figsize=(6, 5))
        y_true = test_stats['all_bin_targets'][:, i]
        y_pred = test_stats['all_bin_preds'][:, i]

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
        plt.title(f'Confusion Matrix: {label_name}')
        plt.tight_layout()

        if save_fig:
            # Create a filename-safe version of the label name
            safe_name = "".join(c if c.isalnum() else "_" for c in label_name)
            fig_path = os.path.join(CHARTS_DIR, f"cm_binary_{safe_name}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix for {label_name} to {fig_path}")

        plt.show()

    # Glass confusion matrix (if available)
    if 'all_glass_targets' in test_stats and 'all_glass_preds' in test_stats:
        mask_valid = test_stats['all_glass_targets'] != IGNORE_GLASS_LABEL
        if mask_valid.sum() > 0:
            fig = plt.figure(figsize=(8, 6))
            y_true = test_stats['all_glass_targets'][mask_valid]
            y_pred = test_stats['all_glass_preds'][mask_valid]

            # Find unique classes that are actually present in the data
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

            # Create labels only for classes that exist in the data
            glass_labels = []
            for class_idx in unique_classes:
                for name, idx in GLASS_MAP.items():
                    if idx == class_idx:
                        glass_labels.append(name)
                        break
                else:
                    glass_labels.append(f"Class {class_idx}")  # Fallback

            # Use the same unique classes for confusion matrix computation
            cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=glass_labels)
            disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
            plt.title('Confusion Matrix: Glass Types')
            plt.tight_layout()

            if save_fig:
                fig_path = os.path.join(CHARTS_DIR, "cm_glass.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Saved glass confusion matrix to {fig_path}")

            plt.show()

# Plot and save confusion matrices
plot_and_save_confusion_matrices(test_stats, save_fig=True)
