# Multi-Label Training Script für "Häuser"-Datensatz
# - Multi-label (6 binäre Merkmale) mit BCEWithLogitsLoss



# Vorbereitung Colab: 
!pip install -q gdown

# Google Drive Datei-ID (ZIP)
file_id = "10mXMyAPXm7t4PmmkKY32YMKpILzjP5NA"
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
import torchvision.models as models

import torchvision.transforms as T


# optional: sklearn für Metriken
try:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
except Exception:
    !pip install -q scikit-learn
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error


# Settings & Mapping
ROOT_DIR = "data/Test_Training_DATA_single"
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



# Dataset
class BinaryOnlyDataset(Dataset):
    def __init__(self, root_dir, label_cols=BINARY_LABEL_COLUMNS, transform=None,
                 image_extensions=('.jpg', '.jpeg', '.png')):
        self.root = root_dir
        self.folders = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.label_cols = label_cols
        self.image_extensions = [ext.lower() for ext in image_extensions]
        self.transform = transform or T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Alle Bilder in den Ordnern finden und indexieren
        self.image_paths = []
        for folder in self.folders:
            folder_images = self._find_all_images(folder)
            if not folder_images:
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
        for f in os.listdir(folder):
            if f.lower().endswith(ext.lower()):
                return os.path.join(folder, f)
        return None

    def __len__(self):
        return sum(len(images) for images in self.image_paths)

    def __getitem__(self, idx):
        # Finde Ordner und lokales Bild
        folder_idx = 0
        local_idx = idx
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
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        # CSV für binäre Labels
        csv_path = self._find_file(folder, ".csv")
        if csv_path is None:
            df = pd.DataFrame()
        else:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except Exception:
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    df = pd.DataFrame()

        # NUR Binary labels verarbeiten
        binary = []
        for col in self.label_cols:
            if col in df.columns and df[col].astype(str).str.contains("checked", case=False, na=False).any():
                binary.append(1.0)
            else:
                binary.append(0.0)
        binary = torch.tensor(binary, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        # Rückgabe: NUR img und binary_labels
        return img, binary
    

# Model (Backbone + binary head)
resnet = models.resnet18(pretrained=True)

class DLM_Binary_ResNet(nn.Module):
    def __init__(self, n_binary_labels=len(BINARY_LABEL_COLUMNS)):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Entfernt die letzte FC-Layer des ResNet
        self.fc = nn.Linear(512, 256)
        self.binary_head = nn.Linear(256, n_binary_labels)

    def forward(self, x):
        x = self.backbone(x)           # [batch, 512, 1, 1]
        x = x.view(x.size(0), -1)      # [batch, 512]
        x = torch.relu(self.fc(x))     # [batch, 256]
        b_logits = self.binary_head(x) # [batch, n_binary_labels]
        return b_logits


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

    # Print results in tabular format
    print("\n===== DATASET EVALUATION =====")
    print(f"Total houses: {len(all_folders)}")

    # Print binary label statistics
    print("\n1. BINARY LABELS:")
    headers = ["Label", "Checked", "Unchecked", "Missing"]
    
    header_format = "{:<30} {:<20} {:<20} {:<20}"
    row_format = "{:<30} {:>8d} ({:>5.1f}%) {:>8d} ({:>5.1f}%) {:>8d} ({:>5.1f}%)"

    print(header_format.format(*headers))
    print("-" * 95)

    for col in BINARY_LABEL_COLUMNS:
        total = sum(binary_counts[col].values())
        
        checked = binary_counts[col]["checked"]
        unchecked = binary_counts[col]["unchecked"]
        missing = binary_counts[col]["missing"]
        
        checked_pct = (checked / total) * 100 if total > 0 else 0
        unchecked_pct = (unchecked / total) * 100 if total > 0 else 0
        missing_pct = (missing / total) * 100 if total > 0 else 0
        
        print(row_format.format(
            col,
            checked, checked_pct,      # ← 3 Argumente für "Checked"
            unchecked, unchecked_pct,  # ← 3 Argumente für "Unchecked"
            missing, missing_pct       # ← 3 Argumente für "Missing"
        ))

    return {
        "total_houses": len(all_folders),
        "binary_counts": binary_counts
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

if len(all_folders) == 1:
    # Nur ein Haus vorhanden: in alle Splits packen
    train_list = all_folders
    val_list = all_folders
    test_list = all_folders
else:
    train_list, val_list, test_list = multi_label_stratified_split(
        all_folders, train_size=0.7, val_size=0.15, test_size=0.15
    )

# temporäre kleine Dataset-Wrapper, damit Listen übergeben werden können
class FoldersDataset(BinaryOnlyDataset):
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
    Shows both absolute counts and percentages for binary labels only.

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

    # Initialize counters for each split (nur für binäre Labels)
    stats = {}
    for split_name in datasets.keys():
        stats[split_name] = {
            "count": 0,
            "binary_counts": {col: {"checked": 0, "unchecked": 0} for col in BINARY_LABEL_COLUMNS}
        }

    # Process each dataset
    for split_name, dataset in datasets.items():
        if split_name == "all":
            continue  # We'll compute 'all' at the end

        print(f"Analyzing {split_name} split ({len(dataset)} instances)...")
        stats[split_name]["count"] = len(dataset)

        # Iterate through dataset and count labels
        for i in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
            _, bin_labels = dataset[i]  # Nur 2 Werte: img, bin_labels

            # Binary labels
            for j, col in enumerate(BINARY_LABEL_COLUMNS):
                if bin_labels[j] > 0.5:  # Checked
                    stats[split_name]["binary_counts"][col]["checked"] += 1
                else:  # Unchecked
                    stats[split_name]["binary_counts"][col]["unchecked"] += 1

    # Compute 'all' by combining stats from all splits
    stats["all"] = {"count": 0, "binary_counts": {col: {"checked": 0, "unchecked": 0} for col in BINARY_LABEL_COLUMNS}}
    stats["all"]["count"] = sum(stats[split]["count"] for split in ["train", "val", "test"])

    # Binary labels
    for col in BINARY_LABEL_COLUMNS:
        stats["all"]["binary_counts"][col]["checked"] = sum(
            stats[split]["binary_counts"][col]["checked"] for split in ["train", "val", "test"]
        )
        stats["all"]["binary_counts"][col]["unchecked"] = sum(
            stats[split]["binary_counts"][col]["unchecked"] for split in ["train", "val", "test"]
        )

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

    return stats

# Example usage:
split_stats = evaluate_dataset_splits(train_ds, val_ds, test_ds)

# Darstellung eines Testbildes
def show_random_train_image_with_title(train_folders, title_col=TITLE_COLUMN):
    ds = FoldersDataset(train_folders)
    idx = random.randint(0, len(ds)-1)

    # unpack 2 values (img, binary) - nicht 4!
    sample = ds[idx]
    if len(sample) == 2:
        img_item, bin_labels = sample 
    else:
        raise RuntimeError(f"Dataset returned {len(sample)} items; erwartet 2 (img, bin_labels). Bitte Klassen neu laden.")

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

    # Nur binäre Labels anzeigen - keine Glass/Heating-Werte
    bin_labels_list = {name: int(val) for name, val in zip(BINARY_LABEL_COLUMNS, bin_labels.cpu().numpy())}

    # Zeige auch Bildpfad und Index-Informationen
    img_filename = os.path.basename(img_path)

    # Erstelle Titel nur mit verfügbaren Informationen
    title_str = f"{title_val}\nBild: {img_filename}\n" + ", ".join([f"{k}:{v}" for k,v in bin_labels_list.items()])
    plt.title(title_str, fontsize=9)
    plt.show()

# Aufruf:
show_random_train_image_with_title(train_list)


# Model, Loss, Optimizer
model = DLM_Binary_ResNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# Training & Validation
def evaluate(model, loader):
    model.eval()
    all_bin_targets = []
    all_bin_preds = []
    losses = []

    with torch.no_grad():
        for imgs, bin_labels in loader:
            imgs = imgs.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE)

            b_logits = model(imgs)
            loss_b = criterion(b_logits, bin_labels)
            losses.append(loss_b.item())

            probs = torch.sigmoid(b_logits).cpu().numpy()
            preds_bin = (probs > 0.5).astype(int)
            all_bin_preds.append(preds_bin)
            all_bin_targets.append(bin_labels.cpu().numpy().astype(int))

    all_bin_targets = np.vstack(all_bin_targets) if all_bin_targets else np.zeros((0, len(BINARY_LABEL_COLUMNS)), dtype=int)
    all_bin_preds = np.vstack(all_bin_preds) if all_bin_preds else np.zeros_like(all_bin_targets)

    prf_bin = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average=None, zero_division=0)
    avg_loss_b = np.mean(losses) if losses else 0.0
    
    
    # 1. Exact Match Accuracy (Subset Accuracy)
    if len(all_bin_targets) > 0:
        exact_match_acc = np.all(all_bin_preds == all_bin_targets, axis=1).mean()
    else:
        exact_match_acc = 0.0
    
    # 2. Hamming Score (Label-wise Accuracy)
    if len(all_bin_targets) > 0:
        hamming_score = (all_bin_preds == all_bin_targets).mean()
    else:
        hamming_score = 0.0
    
    # 3. Per-Label Accuracy
    per_label_acc = []
    for i in range(len(BINARY_LABEL_COLUMNS)):
        if len(all_bin_targets) > 0:
            acc = (all_bin_preds[:, i] == all_bin_targets[:, i]).mean()
        else:
            acc = 0.0
        per_label_acc.append(acc)
    per_label_acc = np.array(per_label_acc)
    
    # 4. Macro-Average Accuracy
    macro_acc = per_label_acc.mean()

    return {
        "binary_prf": prf_bin,
        "loss_b": avg_loss_b,
        "all_bin_targets": all_bin_targets,
        "all_bin_preds": all_bin_preds,
        "exact_match_acc": exact_match_acc,
        "hamming_score": hamming_score,
        "per_label_acc": per_label_acc,
        "macro_acc": macro_acc  # ← DIESE ZEILE FEHLTE!
    }




# Training loop mit Early Stopping
train_losses_epochs = []
val_loss_b_epochs = []
val_exact_match_acc_epochs = []  # ← NEU hinzufügen
val_hamming_score_epochs = []     # ← NEU hinzufügen
val_macro_acc_epochs = []         # ← NEU hinzufügen

# Early Stopping initialisieren
early_stopping = EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True)

# Initialisiere best_val_loss mit einem sehr hohen Wert
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    iters = 0
    for imgs, bin_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"): 
        imgs = imgs.to(DEVICE)
        bin_labels = bin_labels.to(DEVICE)

        optimizer.zero_grad()
        b_logits = model(imgs)
        loss = criterion(b_logits, bin_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iters += 1

    avg_train_loss = running_loss / max(1, iters)
    train_losses_epochs.append(avg_train_loss)  # ← DIESE ZEILE FEHLT!
    
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")

    # Validation
    val_stats = evaluate(model, val_loader)
    avg_val_loss = val_stats['loss_b']
    

    val_loss_b_epochs.append(avg_val_loss)
    val_exact_match_acc_epochs.append(val_stats['exact_match_acc'])  
    val_hamming_score_epochs.append(val_stats['hamming_score'])     
    val_macro_acc_epochs.append(val_stats['macro_acc'])              
    
    print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}")
    print(f"  Exact Match Acc: {val_stats['exact_match_acc']:.3f}")
    print(f"  Hamming Score:   {val_stats['hamming_score']:.3f}")
    print(f"  Macro Accuracy:  {val_stats['macro_acc']:.3f}")
    
    # Binary metrics (wie bisher)
    prec, rec, f1, sup = val_stats['binary_prf']
    print("Binary labels (per class):")
    for i, col in enumerate(BINARY_LABEL_COLUMNS):
        acc = val_stats['per_label_acc'][i]
        print(f"  {col:25s}  P={prec[i]:.3f}  R={rec[i]:.3f}  F1={f1[i]:.3f}  Acc={acc:.3f}  support={sup[i]}")
    
    # Modell-Checkpointing (bestes Modell basierend auf Validation-Loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_housenet.pth")
        print("Saved best model -> best_housenet.pth")

    # Early Stopping prüfen
    if early_stopping(avg_val_loss, model):
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
    ensure_dir(results_dir)
    folders = folders_list or test_loader.dataset.folders

    model.eval()
    all_bin_targets = []
    all_bin_preds = []
    losses = []

    folder_ptr = 0

    with torch.no_grad():
        for imgs, bin_labels in test_loader: 
            batch_size_actual = imgs.size(0)
            batch_folders = folders[folder_ptr: folder_ptr + batch_size_actual]
            folder_ptr += batch_size_actual

            imgs = imgs.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE)

            b_logits = model(imgs)
            loss_b = criterion(b_logits, bin_labels).item()
            losses.append(loss_b)

            probs = torch.sigmoid(b_logits).cpu().numpy()
            preds_bin = (probs > 0.5).astype(int)
            all_bin_preds.append(preds_bin)
            all_bin_targets.append(bin_labels.cpu().numpy().astype(int))

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

                out_csv_path = os.path.join(results_dir, f"{base_name}_prediction.csv")
                with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)

    # concat arrays for aggregated metrics
    all_bin_targets = np.vstack(all_bin_targets) if all_bin_targets else np.zeros((0, len(BINARY_LABEL_COLUMNS)), dtype=int)
    all_bin_preds   = np.vstack(all_bin_preds)   if all_bin_preds   else np.zeros_like(all_bin_targets)

    prf_bin = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average=None, zero_division=0)
    avg_loss_b = np.mean(losses) if losses else 0.0
    
    if len(all_bin_targets) > 0:
        exact_match_acc = np.all(all_bin_preds == all_bin_targets, axis=1).mean()
        hamming_score = (all_bin_preds == all_bin_targets).mean()
        per_label_acc = []
        for i in range(len(BINARY_LABEL_COLUMNS)):
            acc = (all_bin_preds[:, i] == all_bin_targets[:, i]).mean()
            per_label_acc.append(acc)
        per_label_acc = np.array(per_label_acc)
        macro_acc = per_label_acc.mean()
    else:
        exact_match_acc = 0.0
        hamming_score = 0.0
        per_label_acc = np.zeros(len(BINARY_LABEL_COLUMNS))
        macro_acc = 0.0

    return {
        "binary_prf": prf_bin,
        "loss_b": avg_loss_b,
        "all_bin_targets": all_bin_targets,
        "all_bin_preds": all_bin_preds,
        "exact_match_acc": exact_match_acc,
        "hamming_score": hamming_score,
        "per_label_acc": per_label_acc,
        "macro_acc": macro_acc
    }



# Performance Metriken
ensure_dir(CHARTS_DIR)

def plot_binary_training_metrics(train_losses_epochs, val_loss_b_epochs, 
                                 val_exact_match_acc_epochs, val_hamming_score_epochs,
                                 binary_prf, per_label_acc, test_stats, save_fig=True):
    """
    Plots training loss curves, accuracy curves, binary metrics per class, 
    and confusion matrices for binary labels.
    """
    # 1. Training Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(train_losses_epochs)+1), train_losses_epochs, label='Train Loss', color='blue', marker='o')
    plt.plot(np.arange(1, len(val_loss_b_epochs)+1), val_loss_b_epochs, label='Val Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Binary Classification Loss per Epoch')
    plt.legend()
    plt.grid(alpha=0.2)
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "binary_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    if len(val_exact_match_acc_epochs) > 0 and len(val_hamming_score_epochs) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(val_exact_match_acc_epochs)+1), val_exact_match_acc_epochs, 
                 label='Exact Match Acc', color='green', marker='s')
        plt.plot(np.arange(1, len(val_hamming_score_epochs)+1), val_hamming_score_epochs, 
                 label='Hamming Score', color='orange', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Multi-Label Accuracy Metrics per Epoch')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.ylim([0, 1])
        if save_fig:
            plt.savefig(os.path.join(CHARTS_DIR, "binary_accuracy_curve.png"), dpi=300, bbox_inches='tight')
        plt.show()

    # ========== 3. Binary Metrics per Class (INKL. ACCURACY!) ==========
    if binary_prf is not None and len(binary_prf) == 4 and per_label_acc is not None:
        prec, rec, f1, support = binary_prf
        x = np.arange(len(prec))
        width = 0.2  # Schmaler wegen 4 Balken

        label_names = BINARY_LABEL_COLUMNS if len(x) == len(BINARY_LABEL_COLUMNS) else [f"Label {i+1}" for i in range(len(x))]

        plt.figure(figsize=(12, 6))
        plt.bar(x - 1.5*width, prec, width, label='Precision', color='blue')
        plt.bar(x - 0.5*width, rec, width, label='Recall', color='orange')
        plt.bar(x + 0.5*width, f1, width, label='F1', color='green')
        plt.bar(x + 1.5*width, per_label_acc, width, label='Accuracy', color='purple')  
        plt.xlabel('Binary Classes')
        plt.ylabel('Score')
        plt.title('Binary Classification Metrics per Class')
        plt.xticks(x, [label[:12] + '...' if len(label) > 12 else label for label in label_names], rotation=45, ha='right')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.ylim([0, 1.05])  # Etwas Platz oben
        plt.tight_layout()
        if save_fig:
            plt.savefig(os.path.join(CHARTS_DIR, "binary_metrics_per_class.png"), dpi=300, bbox_inches='tight')
        plt.show()

    # 4. Confusion Matrices (unverändert)
    if test_stats is not None:
        for i, label_name in enumerate(BINARY_LABEL_COLUMNS):
            fig = plt.figure(figsize=(6, 5))
            y_true = test_stats['all_bin_targets'][:, i]
            y_pred = test_stats['all_bin_preds'][:, i]
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
            disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
            plt.title(f'Confusion Matrix: {label_name}')
            plt.tight_layout()
            if save_fig:
                safe_name = "".join(c if c.isalnum() else "_" for c in label_name)
                fig_path = os.path.join(CHARTS_DIR, f"cm_binary_{safe_name}.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.show()

# Nach Training und Test aufrufen:
# Testen des Modells und Berechnung der Test-Metriken
test_stats = test_model_and_write_results(model, test_loader)

# Jetzt die Plots erzeugen
plot_binary_training_metrics(
    train_losses_epochs=train_losses_epochs,
    val_loss_b_epochs=val_loss_b_epochs,
    val_exact_match_acc_epochs=val_exact_match_acc_epochs,  
    val_hamming_score_epochs=val_hamming_score_epochs,     
    binary_prf=test_stats['binary_prf'],
    per_label_acc=test_stats['per_label_acc'],              
    test_stats=test_stats,
    save_fig=True
)