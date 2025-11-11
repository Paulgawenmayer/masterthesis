# Regression Training Script für "Häuser"-Datensatz
# - Regression (Heizwärmebedarf) mit MSELoss
# Hinweis: in Colab vorher Google Drive mounten (siehe unten)

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
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:
    !pip install -q scikit-learn
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Dataset
class HeatingOnlyDataset(Dataset):
    def __init__(self, root_dir, heating_log_mean=0.0, heating_log_std=1.0, transform=None, image_extensions=('.jpg', '.jpeg', '.png')):
        self.root = root_dir
        self.heating_log_mean = heating_log_mean
        self.heating_log_std = heating_log_std
        self.folders = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
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

        # CSV für Heizwärmebedarf
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

        # NUR Heizwärmebedarf verarbeiten - MIT LOG-NORMALISIERUNG
        heating_value = float("nan")
        if HEATING_DEMAND in df.columns:
            vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
            found = None
            for v in vals:
                if not pd.isna(v) and v > 0:  # Nur positive Werte
                    found = v
                    break
            if found is not None:
                if USE_LOG_NORMALIZATION:
                    # Log-Normalisierung: (log(x) - log_mean) / log_std
                    log_value = np.log(float(found))
                    heating_value = (log_value - self.heating_log_mean) / self.heating_log_std
                else:
                    # Keine Normalisierung bei Ein-Haus-Datensatz
                    heating_value = float(found) / 10000.0  # Einfache Skalierung: kWh → Zehntausende

        if self.transform:
            img = self.transform(img)

        heating_value = torch.tensor(heating_value, dtype=torch.float32)


        # Rückgabe: NUR img und heating_value
        return img, heating_value


# Model (Backbone + heating head)
resnet = models.resnet18(pretrained=True)

class DLM_Regression_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Entfernt die letzte FC-Layer des ResNet
        self.fc = nn.Linear(512, 256)
        self.heating_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.backbone(x)           # [batch, 512, 1, 1]
        x = x.view(x.size(0), -1)      # [batch, 512]
        x = torch.relu(self.fc(x))     # [batch, 256]
        h_pred = self.heating_head(x).squeeze(1)  # [batch]
        return h_pred


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
    def __init__(self, patience=10, min_delta=0.00001, restore_best_weights=True):
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

all_folders = sorted([os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR)
                    if os.path.isdir(os.path.join(ROOT_DIR, d))])

def evaluate_dataset(root_dir=ROOT_DIR):
    """
    Evaluates the entire dataset and provides statistics about the heating demand.

    Args:
        root_dir: The root directory containing all the house folders

    Returns:
        A dictionary with dataset statistics
    """


    print(f"Analyzing {len(all_folders)} houses in {root_dir}...")

    # Initialize counters
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
            heating_stats["missing"] += 1
            continue

        # Load CSV
        try:
            df = pd.read_csv(csv_path, sep=";")
        except Exception:
            try:
                df = pd.read_csv(csv_path)  # fallback
            except Exception:
                heating_stats["missing"] += 1
                continue

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

    # Print heating demand statistics
    print("\nHEATING DEMAND:")
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
        "heating_stats": heating_stats
    }

dataset_stats = evaluate_dataset(ROOT_DIR)



# Verschiebe diesen Block NACH dataset_stats = evaluate_dataset(ROOT_DIR) (ca. Zeile 350):
def calculate_heating_log_stats(all_folders):
    """Berechnet Mean und Std für Log-Heizwärmebedarf-Normalisierung"""
    heating_values = []
    
    for folder in all_folders:
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break
        
        if csv_path is not None:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except:
                try:
                    df = pd.read_csv(csv_path)
                except:
                    continue
            
            if HEATING_DEMAND in df.columns:
                vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
                for v in vals:
                    if not pd.isna(v) and v > 0:  # Nur positive Werte für Log
                        heating_values.append(float(v))
                        break
    
    if heating_values:
        log_values = np.log(heating_values)
        mean_log = np.mean(log_values)
        std_log = np.std(log_values)
        
        # SPEZIALBEHANDLUNG: Wenn nur ein Wert vorhanden ist
        if len(heating_values) == 1 or std_log == 0.0:
            print(f"⚠️  Nur {len(heating_values)} einzigartige Heizwerte gefunden!")
            print(f"Verwende KEINE Log-Normalisierung (da Std=0)")
            print(f"Original Wert: {heating_values[0]:.0f} kWh, Log-Wert: {mean_log:.3f}")
            return None, None  # Signalisiert: keine Normalisierung
        
        print(f"Log-Heating Normalization Stats: Log-Mean={mean_log:.3f}, Log-Std={std_log:.3f}")
        print(f"Original range: {min(heating_values):.0f} - {max(heating_values):.0f} kWh")
        print(f"Log range: {min(log_values):.3f} - {max(log_values):.3f}")
        return mean_log, std_log
    else:
        return 0.0, 1.0

# Normalisierungsstatistiken berechnen
HEATING_LOG_MEAN, HEATING_LOG_STD = calculate_heating_log_stats(all_folders)

# Prüfe, ob Normalisierung möglich ist
USE_LOG_NORMALIZATION = (HEATING_LOG_MEAN is not None and HEATING_LOG_STD is not None)
if not USE_LOG_NORMALIZATION:
    print("🔄 Fallback: Verwende KEINE Normalisierung (Raw-Werte)")
    HEATING_LOG_MEAN, HEATING_LOG_STD = 0.0, 1.0  # Dummy-Werte

def denormalize_heating_log(normalized_values):
    """Konvertiert normalisierte Werte zurück in kWh"""
    if USE_LOG_NORMALIZATION:
        log_values = normalized_values * HEATING_LOG_STD + HEATING_LOG_MEAN
        return np.exp(log_values)  # exp(log(x)) = x
    else:
        # Einfache Rückskalierung bei Ein-Haus-Fall
        return normalized_values * 10000.0  # Zehntausende → kWh



# Funktion zur stratifizierten Aufteilung
def heating_stratified_split(folders, train_size=0.7, val_size=0.15, test_size=0.15, random_state=SEED):
    """
    Stratifizierte Aufteilung von Ordnern basierend auf Heizwärmebedarf.
    Stellt sicher, dass verschiedene Wertebereiche in allen Splits vertreten sind.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split-Anteile müssen sich zu 1.0 summieren"

    # Sammle Heizwärmebedarf für alle Ordner
    folder_heating = {}
    for folder in folders:
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break
        heating_value = None
        if csv_path is not None:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except:
                try:
                    df = pd.read_csv(csv_path)
                except:
                    df = pd.DataFrame()
            if HEATING_DEMAND in df.columns:
                vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
                for v in vals:
                    if not pd.isna(v):
                        heating_value = float(v)
                        break
        folder_heating[folder] = heating_value

    # Stratifizierungslabels basierend auf Heizwärmebedarf
    strat_labels = []
    folders_list = list(folder_heating.keys())
    
    # Berechne Quartile für verfügbare Werte
    available_values = [v for v in folder_heating.values() if v is not None]
    if available_values:
        q25 = np.percentile(available_values, 25)
        q50 = np.percentile(available_values, 50)
        q75 = np.percentile(available_values, 75)
    
    for folder in folders_list:
        heating_val = folder_heating[folder]
        if heating_val is None:
            strat_group = "missing"
        elif heating_val <= q25:
            strat_group = "low"
        elif heating_val <= q50:
            strat_group = "medium_low"
        elif heating_val <= q75:
            strat_group = "medium_high"
        else:
            strat_group = "high"
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

    # Ausgabe der finalen Aufteilung
    print(f"\nFinale Aufteilung:")
    print(f"  Train: {len(train_list)} Ordner")
    print(f"  Val:   {len(val_list)} Ordner")
    print(f"  Test:  {len(test_list)} Ordner")

    return train_list, val_list, test_list

# Erst JETZT die stratifizierte Aufteilung und Dataset-Erstellung:
if len(all_folders) == 1:
    # Nur ein Haus vorhanden: in alle Splits packen
    train_list = all_folders
    val_list = all_folders
    test_list = all_folders
else:
    train_list, val_list, test_list = heating_stratified_split(
        all_folders, train_size=0.7, val_size=0.15, test_size=0.15
    )

# temporäre kleine Dataset-Wrapper, damit Listen übergeben werden können
class FoldersDataset(HeatingOnlyDataset):
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

# Create all datasets mit Log-Normalisierung:
train_ds = FoldersDataset(train_list, heating_log_mean=HEATING_LOG_MEAN, heating_log_std=HEATING_LOG_STD)
val_ds   = FoldersDataset(val_list, heating_log_mean=HEATING_LOG_MEAN, heating_log_std=HEATING_LOG_STD)
test_ds  = FoldersDataset(test_list, heating_log_mean=HEATING_LOG_MEAN, heating_log_std=HEATING_LOG_STD)

# Create all dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Dataset sizes: Train: {len(train_ds)}, Validation: {len(val_ds)}, Test: {len(test_ds)}")


# Darstellung der Labelverteilung in den Trainingsdaten
def evaluate_dataset_splits(train_ds, val_ds, test_ds):
    """
    Evaluates the heating demand distribution in train, validation and test datasets.
    Shows both absolute counts and percentages for heating demand only.

    Args:
        train_ds, val_ds, test_ds: The dataset objects

    Returns:
        Dictionary with statistics about heating demand distributions in each split
    """
    datasets = {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "all": None
    }

    # Initialize counters for each split (nur für Heizwärmebedarf)
    stats = {}
    for split_name in datasets.keys():
        stats[split_name] = {
            "count": 0,
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
        for i in range(len(dataset)):
            _, heating_value = dataset[i]  # Nur 2 Werte: img, heating_value

            # Heating demand
            if not torch.isnan(heating_value):
                stats[split_name]["heating_available"] += 1
                stats[split_name]["heating_values"].append(float(heating_value.item()))
            else:
                stats[split_name]["heating_missing"] += 1

    # Compute 'all' by combining stats from all splits
    stats["all"] = {
        "count": 0,
        "heating_available": 0,
        "heating_missing": 0,
        "heating_values": []
    }
    stats["all"]["count"] = sum(stats[split]["count"] for split in ["train", "val", "test"])

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

    # Print heating demand statistics
    print("\nHEATING DEMAND:")
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

    # unpack 2 values (img, heating_value) - nicht 4!
    sample = ds[idx]
    if len(sample) == 2:
        img_item, heating_value = sample 
    else:
        raise RuntimeError(f"Dataset returned {len(sample)} items; erwartet 2 (img, heating_value). Bitte Klassen neu laden.")

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

    # Nur Heizwärmebedarf anzeigen
    heating_text = f"Heating: {float(heating_value.cpu().item()):.1f} kWh" if not np.isnan(float(heating_value.cpu().item())) else "Heating: MISSING"

    # Zeige auch Bildpfad und Index-Informationen
    img_filename = os.path.basename(img_path)

    # Erstelle Titel nur mit verfügbaren Informationen
    title_str = f"{title_val}\nBild: {img_filename}\n{heating_text}"
    plt.title(title_str, fontsize=9)
    plt.show()

# Aufruf:
show_random_train_image_with_title(train_list)

# Model, Loss, Optimizer
model = DLM_Regression_ResNet().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training & Validation
def evaluate(model, loader):
    model.eval()
    all_heating_targets = []
    all_heating_preds = []
    losses = []

    with torch.no_grad():
        for imgs, heating_labels in loader:
            imgs = imgs.to(DEVICE)
            heating_labels = heating_labels.to(DEVICE)

            h_preds = model(imgs)

            # heating loss only for valid targets
            mask_heating = ~torch.isnan(heating_labels)
            if mask_heating.any():
                h_pred_sel = h_preds[mask_heating]
                h_target_sel = heating_labels[mask_heating]
                loss_h = criterion(h_pred_sel, h_target_sel)
                losses.append(loss_h.item())

            # heating preds/targets (store NaNs as-is)
            all_heating_preds.append(h_preds.cpu().numpy())
            all_heating_targets.append(heating_labels.cpu().numpy())

    # concat
    all_heating_targets = np.concatenate(all_heating_targets) if all_heating_targets else np.array([], dtype=float)
    all_heating_preds   = np.concatenate(all_heating_preds)   if all_heating_preds   else np.array([], dtype=float)

    # Heating metrics: filter out NaNs und LOG-DENORMALISIEREN
    heating_mask = ~np.isnan(all_heating_targets)
    if heating_mask.sum() > 0:
        # Log-Denormalisierung für interpretierbare Metriken
        targets_denorm = denormalize_heating_log(all_heating_targets[heating_mask])
        preds_denorm = denormalize_heating_log(all_heating_preds[heating_mask])
        
        heating_mae = mean_absolute_error(targets_denorm, preds_denorm)
        heating_rmse = math.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        heating_r2 = r2_score(targets_denorm, preds_denorm)
        
        # Zusätzlich: MAPE (Mean Absolute Percentage Error) - sehr nützlich bei Log-Regression
        heating_mape = np.mean(np.abs((targets_denorm - preds_denorm) / targets_denorm)) * 100
    else:
        heating_mae = None
        heating_rmse = None
        heating_r2 = None
        heating_mape = None

    avg_loss_h = np.mean(losses) if losses else 0.0

    return {
        "loss_h": avg_loss_h,
        "heating_mae": heating_mae,
        "heating_rmse": heating_rmse,
        "heating_r2": heating_r2,
        "heating_mape": heating_mape,  # Neue Metrik
        "all_heating_targets": all_heating_targets,
        "all_heating_preds": all_heating_preds
    }

# Training loop mit Early Stopping
train_losses_epochs = []
val_loss_h_epochs = []
val_heating_mae_epochs = []
val_heating_rmse_epochs = []
val_heating_r2_epochs = []

# Early Stopping initialisieren
early_stopping = EarlyStopping(patience=5, min_delta=0.00001, restore_best_weights=True)

# Initialisiere best_val_loss mit einem sehr hohen Wert
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    iters = 0
    for imgs, heating_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        imgs = imgs.to(DEVICE)
        heating_labels = heating_labels.to(DEVICE)

        optimizer.zero_grad()
        h_preds = model(imgs)

        mask_heating = ~torch.isnan(heating_labels)
        if mask_heating.any():
            h_pred_sel = h_preds[mask_heating]
            h_target_sel = heating_labels[mask_heating]
            loss = criterion(h_pred_sel, h_target_sel)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iters += 1

    avg_train_loss = running_loss / max(1, iters)
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")

    # Validation
    val_stats = evaluate(model, val_loader)
    avg_val_loss = val_stats['loss_h']
    print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}")

    # Metriken speichern für Plot
    train_losses_epochs.append(avg_train_loss)
    val_loss_h_epochs.append(avg_val_loss)
    val_heating_mae_epochs.append(val_stats['heating_mae'] if val_stats['heating_mae'] is not None else np.nan)
    val_heating_rmse_epochs.append(val_stats['heating_rmse'] if val_stats['heating_rmse'] is not None else np.nan)
    val_heating_r2_epochs.append(val_stats['heating_r2'] if val_stats['heating_r2'] is not None else np.nan)

    # Heating metrics
    if val_stats['heating_mae'] is not None:
        print(f"Heating MAE: {val_stats['heating_mae']:.2f} kWh")
        print(f"Heating RMSE: {val_stats['heating_rmse']:.2f} kWh")
        print(f"Heating R²: {val_stats['heating_r2']:.3f}")
    else:
        print("No valid heating values in validation set (skipping heating metrics)")

    # Modell-Checkpointing (bestes Modell basierend auf Validation-Loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_heatingnet.pth")
        print("Saved best model -> best_heatingnet.pth")

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
    all_heating_targets = []
    all_heating_preds = []
    losses = []

    folder_ptr = 0

    with torch.no_grad():
        for imgs, heating_labels in test_loader:
            batch_size_actual = imgs.size(0)
            batch_folders = folders[folder_ptr: folder_ptr + batch_size_actual]
            folder_ptr += batch_size_actual

            imgs = imgs.to(DEVICE)
            heating_labels = heating_labels.to(DEVICE)

            h_preds = model(imgs)

            mask_heating = ~torch.isnan(heating_labels)
            if mask_heating.any():
                h_pred_sel = h_preds[mask_heating]
                h_target_sel = heating_labels[mask_heating]
                loss_h = criterion(h_pred_sel, h_target_sel).item()
                losses.append(loss_h)

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

                # heating prediction and true value
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
    all_heating_targets = np.concatenate(all_heating_targets) if all_heating_targets else np.array([], dtype=float)
    all_heating_preds   = np.concatenate(all_heating_preds)   if all_heating_preds   else np.array([], dtype=float)

    # Heating metrics mit Log-Denormalisierung
    heating_mask = ~np.isnan(all_heating_targets)
    if heating_mask.sum() > 0:
        # Log-Denormalisierung für interpretierbare Metriken
        targets_denorm = denormalize_heating_log(all_heating_targets[heating_mask])
        preds_denorm = denormalize_heating_log(all_heating_preds[heating_mask])
        
        heating_mae = mean_absolute_error(targets_denorm, preds_denorm)
        heating_rmse = math.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        heating_r2 = r2_score(targets_denorm, preds_denorm)
    else:
        heating_mae = None
        heating_rmse = None
        heating_r2 = None

    avg_loss_h = np.mean(losses) if losses else 0.0

    return {
        "loss_h": avg_loss_h,
        "heating_mae": heating_mae,
        "heating_rmse": heating_rmse,
        "heating_r2": heating_r2,
        "all_heating_targets": all_heating_targets,
        "all_heating_preds": all_heating_preds
    }

# Performance Metriken
ensure_dir(CHARTS_DIR)

def plot_heating_training_metrics(train_losses_epochs, val_loss_h_epochs, val_heating_mae_epochs, 
                                 val_heating_rmse_epochs, val_heating_r2_epochs, test_stats, save_fig=True):
    """
    Plots training loss curves, heating metrics over epochs, and prediction scatter plots for heating regression.
    """
    # 1. Training Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(train_losses_epochs)+1), train_losses_epochs, label='Train Loss', color='blue', marker='o')
    plt.plot(np.arange(1, len(val_loss_h_epochs)+1), val_loss_h_epochs, label='Val Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Heating Regression Loss per Epoch')
    plt.legend()
    plt.grid(alpha=0.2)
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "heating_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. MAE and RMSE over epochs
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # MAE
    epochs = np.arange(1, len(val_heating_mae_epochs)+1)
    valid_mask = ~np.isnan(val_heating_mae_epochs)
    if np.any(valid_mask):
        ax1.plot(epochs[valid_mask], np.array(val_heating_mae_epochs)[valid_mask], color='purple', marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MAE (kWh)')
        ax1.set_title('Heating Mean Absolute Error')
        ax1.grid(alpha=0.2)

    # RMSE
    valid_mask = ~np.isnan(val_heating_rmse_epochs)
    if np.any(valid_mask):
        ax2.plot(epochs[valid_mask], np.array(val_heating_rmse_epochs)[valid_mask], color='orange', marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE (kWh)')
        ax2.set_title('Heating Root Mean Squared Error')
        ax2.grid(alpha=0.2)

    # R²
    valid_mask = ~np.isnan(val_heating_r2_epochs)
    if np.any(valid_mask):
        ax3.plot(epochs[valid_mask], np.array(val_heating_r2_epochs)[valid_mask], color='green', marker='o')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('R²')
        ax3.set_title('R² Score')
        ax3.grid(alpha=0.2)
        ax3.set_ylim([-1, 1])

    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "heating_metrics_curves.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Prediction vs True Values Scatter Plot
    if test_stats is not None and 'all_heating_targets' in test_stats and 'all_heating_preds' in test_stats:
        heating_mask = ~np.isnan(test_stats['all_heating_targets'])
        if heating_mask.sum() > 0:
            y_true = test_stats['all_heating_targets'][heating_mask]
            y_pred = test_stats['all_heating_preds'][heating_mask]

            plt.figure(figsize=(8, 8))
            plt.scatter(y_true, y_pred, alpha=0.6, color='purple')
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('True Heating Demand (kWh)')
            plt.ylabel('Predicted Heating Demand (kWh)')
            plt.title('Heating Demand: Predictions vs True Values')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Add metrics as text
            if test_stats['heating_mae'] is not None:
                plt.text(0.05, 0.95, f"MAE: {test_stats['heating_mae']:.2f} kWh\nRMSE: {test_stats['heating_rmse']:.2f} kWh\nR²: {test_stats['heating_r2']:.3f}", 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            if save_fig:
                plt.savefig(os.path.join(CHARTS_DIR, "heating_predictions_scatter.png"), dpi=300, bbox_inches='tight')
            plt.show()

            # 4. Residual Plot
            residuals = y_pred - y_true
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.6, color='purple')
            plt.axhline(y=0, color='r', linestyle='--', lw=2)
            plt.xlabel('Predicted Heating Demand (kWh)')
            plt.ylabel('Residuals (Predicted - True)')
            plt.title('Residual Plot: Heating Demand Predictions')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            if save_fig:
                plt.savefig(os.path.join(CHARTS_DIR, "heating_residuals.png"), dpi=300, bbox_inches='tight')
            plt.show()

# Nach Training und Test aufrufen:
# Testen des Modells und Berechnung der Test-Metriken
test_stats = test_model_and_write_results(model, test_loader)

print("Test Loss H:{:.4f}".format(test_stats['loss_h']))
if test_stats['heating_mae'] is not None:
    print(f"Heating MAE: {test_stats['heating_mae']:.2f} kWh")
    print(f"Heating RMSE: {test_stats['heating_rmse']:.2f} kWh")
    print(f"Heating R²: {test_stats['heating_r2']:.3f}")

print("Ergebnisse geschrieben nach:", RESULTS_DIR)

# Jetzt die Plots erzeugen
plot_heating_training_metrics(
    train_losses_epochs=train_losses_epochs,
    val_loss_h_epochs=val_loss_h_epochs,
    val_heating_mae_epochs=val_heating_mae_epochs,
    val_heating_rmse_epochs=val_heating_rmse_epochs,
    val_heating_r2_epochs=val_heating_r2_epochs,
    test_stats=test_stats,
    save_fig=True
)
