# Multi-Task Training Script für "Häuser"-Datensatz
# - Multi-label (6 binäre Merkmale) mit BCEWithLogitsLoss
# - Multi-class (Verglasungstyp: 3 Klassen) mit CrossEntropyLoss
# - Regression (Heizwärmebedarf) mit MSELoss

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

# sklearn für Metriken
try:
    from sklearn.metrics import (precision_recall_fscore_support, accuracy_score, 
                                confusion_matrix, ConfusionMatrixDisplay, 
                                mean_absolute_error, mean_squared_error, r2_score)
except Exception:
    !pip install -q scikit-learn
    from sklearn.metrics import (precision_recall_fscore_support, accuracy_score, 
                                confusion_matrix, ConfusionMatrixDisplay, 
                                mean_absolute_error, mean_squared_error, r2_score)

# Settings
ROOT_DIR = "data/Test_Training_DATA_single"
RESULTS_DIR = "DLM_predictions_multitask"
CHARTS_DIR = "Performance_Metrics_Multitask"
print("ROOT_DIR gesetzt auf:", ROOT_DIR)

BATCH_SIZE = 16
IMG_SIZE = 224
NUM_EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Spaltennamen aus CSV
TITLE_COLUMN = "Adresse"
HEATING_DEMAND = "Jahresbedarf [kWh]"
BINARY_LABEL_COLUMNS = [
    "Aufsparrendämmung?",
    "Dach gedämmt?",
    "Dach saniert?",
    "Fassadendämmung",
    "Sockeldämmung",
    "Fenster fassadenbündig"
]
GLASS_COL = "Verglasungstyp"
GLASS_MAP = {"Einfachverglasung": 0, "Zweifachverglasung": 1, "Dreifachverglasung": 2}
IGNORE_GLASS_LABEL = -100

# Helper functions
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_title_from_folder(folder, title_col=TITLE_COLUMN):
    """Liest den Titel (Adresse) aus der CSV im Ordner."""
    csv_path = None
    for f in os.listdir(folder):
        if f.lower().endswith(".csv"):
            csv_path = os.path.join(folder, f)
            break
    if csv_path is None:
        return None
    try:
        df = pd.read_csv(csv_path, sep=";")
    except:
        try:
            df = pd.read_csv(csv_path)
        except:
            return None
    if title_col in df.columns:
        val = df[title_col].values[0]
        if pd.notna(val):
            return str(val)
    return None

def calculate_heating_log_stats(all_folders):
    heating_values = []
    for folder in all_folders:
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break
        if csv_path:
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
                    if not pd.isna(v) and v > 0:
                        heating_values.append(float(v))
                        break
    
    if heating_values and len(heating_values) > 1:
        log_values = np.log(heating_values)
        mean_log = np.mean(log_values)
        std_log = np.std(log_values)
        if std_log > 0:
            print(f"Log-Heating Stats: Mean={mean_log:.3f}, Std={std_log:.3f}")
            return mean_log, std_log
    return 0.0, 1.0

def denormalize_heating_log(normalized_values, mean, std):
    log_values = normalized_values * std + mean
    return np.exp(log_values)

# PUNKT 1: Stratified Split 
def heating_stratified_split(all_folders, test_frac=0.15, val_frac=0.15, random_state=42):
    """
    Stratifizierter Split basierend auf Heizwärmebedarf-Quartilen.
    """
    heating_values = []
    for folder in all_folders:
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break
        if csv_path:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except:
                try:
                    df = pd.read_csv(csv_path)
                except:
                    heating_values.append(float('nan'))
                    continue
            if HEATING_DEMAND in df.columns:
                vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
                found = False
                for v in vals:
                    if not pd.isna(v) and v > 0:
                        heating_values.append(float(v))
                        found = True
                        break
                if not found:
                    heating_values.append(float('nan'))
            else:
                heating_values.append(float('nan'))
        else:
            heating_values.append(float('nan'))
    
    # Stratifizierungs-Labels erstellen
    heating_values = np.array(heating_values)
    valid_mask = ~np.isnan(heating_values)
    
    if valid_mask.sum() > 1:
        valid_vals = heating_values[valid_mask]
        q25, q50, q75 = np.percentile(valid_vals, [25, 50, 75])
        
        strat_labels = []
        for h in heating_values:
            if np.isnan(h):
                strat_labels.append("missing")
            elif h < q25:
                strat_labels.append("low")
            elif h < q50:
                strat_labels.append("medium_low")
            elif h < q75:
                strat_labels.append("medium_high")
            else:
                strat_labels.append("high")
        
        # Stratifizierter Split
        train_folders, test_folders = train_test_split(
            all_folders, test_size=test_frac, stratify=strat_labels, random_state=random_state
        )
        
        # Validation split (auch stratifiziert)
        train_heating = []
        for folder in train_folders:
            idx = all_folders.index(folder)
            train_heating.append(heating_values[idx])
        
        train_strat = []
        for h in train_heating:
            if np.isnan(h):
                train_strat.append("missing")
            elif h < q25:
                train_strat.append("low")
            elif h < q50:
                train_strat.append("medium_low")
            elif h < q75:
                train_strat.append("medium_high")
            else:
                train_strat.append("high")
        
        val_frac_adjusted = val_frac / (1 - test_frac)
        train_folders, val_folders = train_test_split(
            train_folders, test_size=val_frac_adjusted, stratify=train_strat, random_state=random_state
        )
        
        return train_folders, val_folders, test_folders
    else:
        # Fallback: einfacher Random Split
        print("Warnung: Nicht genug Daten für stratifizierten Split. Verwende Random Split.")
        train_folders, test_folders = train_test_split(all_folders, test_size=test_frac, random_state=random_state)
        val_frac_adjusted = val_frac / (1 - test_frac)
        train_folders, val_folders = train_test_split(train_folders, test_size=val_frac_adjusted, random_state=random_state)
        return train_folders, val_folders, test_folders

# PUNKT 2: Detaillierte Dataset-Analyse-Funktionen 

def evaluate_dataset(folders_list, dataset_name="Dataset"):
    """
    Analysiert den Datensatz und gibt Statistiken aus.
    """
    print(f"\n===== {dataset_name} Analyse =====")
    print(f"Anzahl Ordner: {len(folders_list)}")
    
    # Binary Labels
    binary_counts = {col: 0 for col in BINARY_LABEL_COLUMNS}
    
    # Glass Labels
    glass_counts = {"Einfachverglasung": 0, "Zweifachverglasung": 0, "Dreifachverglasung": 0, "Missing": 0}
    
    # Heating values
    heating_values = []
    heating_missing = 0
    
    # Image counts
    total_images = 0
    
    for folder in folders_list:
        # Bilder zählen
        images = []
        for f in os.listdir(folder):
            if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                images.append(f)
        total_images += len(images)
        
        # CSV laden
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break
        
        if csv_path is None:
            heating_missing += 1
            glass_counts["Missing"] += 1
            continue
        
        try:
            df = pd.read_csv(csv_path, sep=";")
        except:
            try:
                df = pd.read_csv(csv_path)
            except:
                heating_missing += 1
                glass_counts["Missing"] += 1
                continue
        
        # Binary Labels
        for col in BINARY_LABEL_COLUMNS:
            if col in df.columns and df[col].astype(str).str.contains("checked", case=False, na=False).any():
                binary_counts[col] += 1
        
        # Glass Label
        if GLASS_COL in df.columns:
            vals = df[GLASS_COL].astype(str).values
            found = False
            for v in vals:
                v_str = str(v).strip()
                if v_str != "" and v_str.lower() != "nan":
                    if v_str in glass_counts:
                        glass_counts[v_str] += 1
                        found = True
                        break
            if not found:
                glass_counts["Missing"] += 1
        else:
            glass_counts["Missing"] += 1
        
        # Heating
        if HEATING_DEMAND in df.columns:
            vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
            found = False
            for v in vals:
                if not pd.isna(v) and v > 0:
                    heating_values.append(float(v))
                    found = True
                    break
            if not found:
                heating_missing += 1
        else:
            heating_missing += 1
    
    # Ausgabe
    print(f"Gesamtbilder: {total_images}")
    print(f"Durchschnittliche Bilder pro Ordner: {total_images/len(folders_list):.1f}")
    
    print("\n--- Binary Labels ---")
    for col, count in binary_counts.items():
        percentage = (count / len(folders_list)) * 100
        print(f"{col}: {count} ({percentage:.1f}%)")
    
    print("\n--- Glass Labels ---")
    for glass_type, count in glass_counts.items():
        percentage = (count / len(folders_list)) * 100
        print(f"{glass_type}: {count} ({percentage:.1f}%)")
    
    print("\n--- Heating Demand ---")
    print(f"Verfügbar: {len(heating_values)} ({(len(heating_values)/len(folders_list))*100:.1f}%)")
    print(f"Fehlend: {heating_missing} ({(heating_missing/len(folders_list))*100:.1f}%)")
    
    if heating_values:
        print(f"Min: {min(heating_values):.0f} kWh")
        print(f"Max: {max(heating_values):.0f} kWh")
        print(f"Mittelwert: {np.mean(heating_values):.0f} kWh")
        print(f"Median: {np.median(heating_values):.0f} kWh")
        print(f"Standardabweichung: {np.std(heating_values):.0f} kWh")
    
    return {
        'binary_counts': binary_counts,
        'glass_counts': glass_counts,
        'heating_values': heating_values,
        'heating_missing': heating_missing,
        'total_images': total_images
    }


def evaluate_dataset_splits(train_list, val_list, test_list):
    """
    Vergleicht die Verteilung über Train/Val/Test Splits.
    """
    print("\n" + "="*60)
    print("DATASET SPLIT ANALYSE")
    print("="*60)
    
    train_stats = evaluate_dataset(train_list, "Training Set")
    val_stats = evaluate_dataset(val_list, "Validation Set")
    test_stats = evaluate_dataset(test_list, "Test Set")
    
    # Vergleichsplot: Binary Labels
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(BINARY_LABEL_COLUMNS))
    width = 0.25
    
    train_counts = [train_stats['binary_counts'][col] for col in BINARY_LABEL_COLUMNS]
    val_counts = [val_stats['binary_counts'][col] for col in BINARY_LABEL_COLUMNS]
    test_counts = [test_stats['binary_counts'][col] for col in BINARY_LABEL_COLUMNS]
    
    ax.bar(x - width, train_counts, width, label='Train', color='skyblue')
    ax.bar(x, val_counts, width, label='Val', color='lightgreen')
    ax.bar(x + width, test_counts, width, label='Test', color='salmon')
    
    ax.set_xlabel('Binary Label')
    ax.set_ylabel('Anzahl')
    ax.set_title('Verteilung Binary Labels über Train/Val/Test')
    ax.set_xticks(x)
    ax.set_xticklabels(BINARY_LABEL_COLUMNS, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    ensure_dir(CHARTS_DIR)
    plt.savefig(os.path.join(CHARTS_DIR, "dataset_binary_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Vergleichsplot: Glass Labels
    fig, ax = plt.subplots(figsize=(10, 6))
    glass_types = ["Einfachverglasung", "Zweifachverglasung", "Dreifachverglasung", "Missing"]
    x = np.arange(len(glass_types))
    
    train_glass = [train_stats['glass_counts'][g] for g in glass_types]
    val_glass = [val_stats['glass_counts'][g] for g in glass_types]
    test_glass = [test_stats['glass_counts'][g] for g in glass_types]
    
    ax.bar(x - width, train_glass, width, label='Train', color='skyblue')
    ax.bar(x, val_glass, width, label='Val', color='lightgreen')
    ax.bar(x + width, test_glass, width, label='Test', color='salmon')
    
    ax.set_xlabel('Verglasungstyp')
    ax.set_ylabel('Anzahl')
    ax.set_title('Verteilung Verglasungstypen über Train/Val/Test')
    ax.set_xticks(x)
    ax.set_xticklabels(glass_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "dataset_glass_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Vergleichsplot: Heating Distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (stats, name, color) in enumerate([
        (train_stats, 'Train', 'skyblue'),
        (val_stats, 'Val', 'lightgreen'),
        (test_stats, 'Test', 'salmon')
    ]):
        if stats['heating_values']:
            axes[idx].hist(stats['heating_values'], bins=20, color=color, alpha=0.7, edgecolor='black')
            axes[idx].set_xlabel('Heizwärmebedarf (kWh)')
            axes[idx].set_ylabel('Häufigkeit')
            axes[idx].set_title(f'{name} Set - Heizwärmebedarf')
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].axvline(np.mean(stats['heating_values']), color='red', 
                            linestyle='--', linewidth=2, label=f'Mean: {np.mean(stats["heating_values"]):.0f}')
            axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "dataset_heating_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("Dataset-Analyse abgeschlossen. Plots gespeichert.")
    print("="*60 + "\n")


def show_random_images_with_labels(folders_list, n=6):
    """
    Zeigt zufällige Beispielbilder mit ihren Labels.
    """
    ensure_dir(CHARTS_DIR)
    
    # Wähle zufällige Ordner
    random_folders = random.sample(folders_list, min(n, len(folders_list)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, folder in enumerate(random_folders):
        # Finde ein Bild
        images = []
        for f in os.listdir(folder):
            if any(f.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                images.append(os.path.join(folder, f))
        
        if not images:
            continue
        
        img_path = random.choice(images)
        
        # Lade CSV
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break
        
        # Sammle Labels
        binary_labels = []
        glass_label = "N/A"
        heating_label = "N/A"
        title = os.path.basename(folder)
        
        if csv_path:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except:
                try:
                    df = pd.read_csv(csv_path)
                except:
                    df = pd.DataFrame()
            
            # Title
            if TITLE_COLUMN in df.columns:
                val = df[TITLE_COLUMN].values[0]
                if pd.notna(val):
                    title = str(val)[:30]  # Kürze auf 30 Zeichen
            
            # Binary
            for col in BINARY_LABEL_COLUMNS:
                if col in df.columns and df[col].astype(str).str.contains("checked", case=False, na=False).any():
                    binary_labels.append(col.split('?')[0][:15])  # Kürze Label
            
            # Glass
            if GLASS_COL in df.columns:
                vals = df[GLASS_COL].astype(str).values
                for v in vals:
                    v_str = str(v).strip()
                    if v_str != "" and v_str.lower() != "nan" and v_str in GLASS_MAP:
                        glass_label = v_str
                        break
            
            # Heating
            if HEATING_DEMAND in df.columns:
                vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
                for v in vals:
                    if not pd.isna(v) and v > 0:
                        heating_label = f"{float(v):.0f} kWh"
                        break
        
        # Zeige Bild
        try:
            img = Image.open(img_path).convert("RGB")
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            # Label-Text
            label_text = f"{title}\n"
            label_text += f"Glass: {glass_label}\n"
            label_text += f"Heating: {heating_label}\n"
            if binary_labels:
                label_text += f"Binary: {', '.join(binary_labels[:2])}"  # Zeige max 2
            else:
                label_text += "Binary: Keine"
            
            axes[idx].set_title(label_text, fontsize=8, ha='left')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error loading\n{os.path.basename(img_path)}", 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    # Leere übrige Subplots
    for idx in range(len(random_folders), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "example_images_with_labels.png"), dpi=300, bbox_inches='tight')
    plt.show()

# Multi-Task Dataset
class MultiTaskDataset(Dataset):
    def __init__(self, root_dir, heating_log_mean=0.0, heating_log_std=1.0, 
                 label_cols=BINARY_LABEL_COLUMNS, glass_col=GLASS_COL,
                 transform=None, image_extensions=('.jpg', '.jpeg', '.png')):
        self.root = root_dir
        self.heating_log_mean = heating_log_mean
        self.heating_log_std = heating_log_std
        self.label_cols = label_cols
        self.glass_col = glass_col
        self.folders = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.image_extensions = [ext.lower() for ext in image_extensions]
        self.transform = transform or T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Bilder indexieren
        self.image_paths = []
        for folder in self.folders:
            folder_images = self._find_all_images(folder)
            if not folder_images:
                print(f"Warnung: Keine Bilder in {folder} gefunden!")
            self.image_paths.append(folder_images)

    def _find_all_images(self, folder):
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
        # Finde Ordner und Bild
        folder_idx = 0
        local_idx = idx
        while local_idx >= len(self.image_paths[folder_idx]):
            local_idx -= len(self.image_paths[folder_idx])
            folder_idx += 1

        folder = self.folders[folder_idx]
        img_path = self.image_paths[folder_idx][local_idx]

        # Bild laden
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Fehler beim Laden von {img_path}: {e}")
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=0)

        # CSV laden
        csv_path = self._find_file(folder, ".csv")
        if csv_path is None:
            df = pd.DataFrame()
        else:
            try:
                df = pd.read_csv(csv_path, sep=";")
            except:
                try:
                    df = pd.read_csv(csv_path)
                except:
                    df = pd.DataFrame()

        # Binary labels
        binary = []
        for col in self.label_cols:
            if col in df.columns and df[col].astype(str).str.contains("checked", case=False, na=False).any():
                binary.append(1.0)
            else:
                binary.append(0.0)
        binary = torch.tensor(binary, dtype=torch.float32)

        # Glass label
        glass_label = IGNORE_GLASS_LABEL
        if self.glass_col in df.columns:
            vals = df[self.glass_col].astype(str).values
            for v in vals:
                v_str = str(v).strip()
                if v_str != "" and v_str.lower() != "nan":
                    if v_str in GLASS_MAP:
                        glass_label = GLASS_MAP[v_str]
                        break
        glass_label = torch.tensor(glass_label, dtype=torch.long)

        # Heating demand mit Log-Normalisierung
        heating_value = float("nan")
        if HEATING_DEMAND in df.columns:
            vals = pd.to_numeric(df[HEATING_DEMAND], errors='coerce').values
            for v in vals:
                if not pd.isna(v) and v > 0:
                    log_value = np.log(float(v))
                    heating_value = (log_value - self.heating_log_mean) / self.heating_log_std
                    break
        heating_value = torch.tensor(heating_value, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, binary, glass_label, heating_value


# Multi-Task Model
class DLM_MultiTask_ResNet(nn.Module):
    def __init__(self, n_binary_labels=len(BINARY_LABEL_COLUMNS), n_glass_types=len(GLASS_MAP)):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Shared FC layer
        self.fc = nn.Linear(512, 256)
        
        # Task-specific heads
        self.binary_head = nn.Linear(256, n_binary_labels)
        self.glass_head = nn.Linear(256, n_glass_types)
        self.heating_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.backbone(x)  # [batch, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 512]
        x = torch.relu(self.fc(x))  # [batch, 256]
        
        binary_logits = self.binary_head(x)
        glass_logits = self.glass_head(x)
        heating_pred = self.heating_head(x).squeeze(1)
        
        return binary_logits, glass_logits, heating_pred


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_val_loss - val_loss > self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            print(f"EarlyStopping: Keine Verbesserung seit {self.counter} Epochen.")
            if self.counter >= self.patience:
                print(f"EarlyStopping: Training wird nach {self.patience} Epochen ohne Verbesserung beendet.")
                self.early_stop = True
        return self.early_stop

    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict({k: v.to(next(model.parameters()).device) for k, v in self.best_weights.items()})
            print("EarlyStopping: Beste Gewichte wiederhergestellt.")


# Evaluation function
def evaluate(model, loader):
    model.eval()
    all_bin_targets, all_bin_preds = [], []
    all_glass_targets, all_glass_preds = [], []
    all_heating_targets, all_heating_preds = [], []
    losses = {'binary': [], 'glass': [], 'heating': [], 'total': []}

    with torch.no_grad():
        for imgs, bin_labels, glass_labels, heating_labels in loader:
            imgs = imgs.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE)
            glass_labels = glass_labels.to(DEVICE)
            heating_labels = heating_labels.to(DEVICE)

            b_logits, g_logits, h_preds = model(imgs)

            # Binary loss
            loss_b = criterion_binary(b_logits, bin_labels)
            losses['binary'].append(loss_b.item())

            # Glass loss
            if (glass_labels != IGNORE_GLASS_LABEL).any():
                loss_g = criterion_glass(g_logits, glass_labels)
                losses['glass'].append(loss_g.item())
            else:
                loss_g = torch.tensor(0.0, device=DEVICE)

            # Heating loss
            mask_heating = ~torch.isnan(heating_labels)
            if mask_heating.any():
                loss_h = criterion_heating(h_preds[mask_heating], heating_labels[mask_heating])
                losses['heating'].append(loss_h.item())
            else:
                loss_h = torch.tensor(0.0, device=DEVICE)

            # Total loss
            total_loss = (LOSS_WEIGHTS['binary'] * loss_b + 
                         LOSS_WEIGHTS['glass'] * loss_g + 
                         LOSS_WEIGHTS['heating'] * loss_h)
            losses['total'].append(total_loss.item())

            # Collect predictions
            all_bin_preds.append((torch.sigmoid(b_logits) > 0.5).cpu().numpy().astype(int))
            all_bin_targets.append(bin_labels.cpu().numpy().astype(int))
            
            all_glass_preds.append(np.argmax(g_logits.cpu().numpy(), axis=1))
            all_glass_targets.append(glass_labels.cpu().numpy())
            
            all_heating_preds.append(h_preds.cpu().numpy())
            all_heating_targets.append(heating_labels.cpu().numpy())

    # Concatenate
    all_bin_targets = np.vstack(all_bin_targets) if all_bin_targets else np.array([])
    all_bin_preds = np.vstack(all_bin_preds) if all_bin_preds else np.array([])
    all_glass_targets = np.concatenate(all_glass_targets) if all_glass_targets else np.array([])
    all_glass_preds = np.concatenate(all_glass_preds) if all_glass_preds else np.array([])
    all_heating_targets = np.concatenate(all_heating_targets) if all_heating_targets else np.array([])
    all_heating_preds = np.concatenate(all_heating_preds) if all_heating_preds else np.array([])

    # Compute metrics
    # Binary
    if all_bin_targets.size > 0:
        prf_bin = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average=None, zero_division=0)
    else:
        prf_bin = None

    # Glass
    mask_valid_glass = all_glass_targets != IGNORE_GLASS_LABEL
    if mask_valid_glass.sum() > 0:
        glass_acc = accuracy_score(all_glass_targets[mask_valid_glass], all_glass_preds[mask_valid_glass])
        prf_glass = precision_recall_fscore_support(
            all_glass_targets[mask_valid_glass], 
            all_glass_preds[mask_valid_glass], 
            average=None, zero_division=0
        )
    else:
        glass_acc = None
        prf_glass = None

    # Heating
    heating_mask = ~np.isnan(all_heating_targets)
    if heating_mask.sum() > 0:
        targets_denorm = denormalize_heating_log(all_heating_targets[heating_mask], HEATING_LOG_MEAN, HEATING_LOG_STD)
        preds_denorm = denormalize_heating_log(all_heating_preds[heating_mask], HEATING_LOG_MEAN, HEATING_LOG_STD)
        heating_mae = mean_absolute_error(targets_denorm, preds_denorm)
        heating_rmse = math.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        heating_r2 = r2_score(targets_denorm, preds_denorm)
    else:
        heating_mae = heating_rmse = heating_r2 = None

    return {
        'losses': {k: np.mean(v) if v else 0.0 for k, v in losses.items()},
        'binary_prf': prf_bin,
        'glass_acc': glass_acc,
        'glass_prf': prf_glass,
        'heating_mae': heating_mae,
        'heating_rmse': heating_rmse,
        'heating_r2': heating_r2,
        'all_bin_targets': all_bin_targets,
        'all_bin_preds': all_bin_preds,
        'all_glass_targets': all_glass_targets,
        'all_glass_preds': all_glass_preds,
        'all_heating_targets': all_heating_targets,
        'all_heating_preds': all_heating_preds
    }


# PUNKT 3: Detaillierte Plotting-Funktionen
def plot_multitask_training_metrics(train_losses, val_losses, val_metrics, save_fig=True):
    """
    Erstellt detaillierte Plots für alle Tasks.
    """
    ensure_dir(CHARTS_DIR)
    
    # 1. Loss-Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total Loss
    axes[0, 0].plot(train_losses, label='Train Total Loss', marker='o')
    axes[0, 0].plot(val_losses['total'], label='Val Total Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Binary Loss
    axes[0, 1].plot(val_losses['binary'], label='Val Binary Loss', marker='s', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Binary Classification Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Glass Loss
    axes[1, 0].plot(val_losses['glass'], label='Val Glass Loss', marker='s', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Glass Classification Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Heating Loss
    axes[1, 1].plot(val_losses['heating'], label='Val Heating Loss', marker='s', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Heating Regression Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "multitask_losses.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Metrics Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Glass Accuracy
    axes[0].plot(val_metrics['glass_acc'], marker='o', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Glass Classification Accuracy')
    axes[0].grid(alpha=0.3)
    
    # Heating MAE
    axes[1].plot(val_metrics['heating_mae'], marker='o', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE (kWh)')
    axes[1].set_title('Heating MAE')
    axes[1].grid(alpha=0.3)
    
    # Heating R²
    axes[2].plot(val_metrics['heating_r2'], marker='o', color='purple')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('R²')
    axes[2].set_title('Heating R²')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "multitask_metrics.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix_glass(y_true, y_pred, save_fig=True):
    """Confusion Matrix für Glass Classification."""
    mask = y_true != IGNORE_GLASS_LABEL
    if mask.sum() == 0:
        print("Keine validen Glass-Labels für Confusion Matrix.")
        return
    
    cm = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Einfach', 'Zweifach', 'Dreifach'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title('Confusion Matrix: Verglasungstyp')
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "glass_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_binary_f1_scores(prf_bin, save_fig=True):
    """F1-Scores pro Binary Label."""
    if prf_bin is None:
        return
    
    f1_scores = prf_bin[2]  # F1 ist der 3. Wert
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(BINARY_LABEL_COLUMNS))
    ax.bar(x, f1_scores, color='skyblue')
    ax.set_xlabel('Binary Label')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Scores pro Binary Label')
    ax.set_xticks(x)
    ax.set_xticklabels(BINARY_LABEL_COLUMNS, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "binary_f1_scores.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_heating_scatter(y_true, y_pred, save_fig=True):
    """Scatter Plot für Heating Prediction."""
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return
    
    y_true_denorm = denormalize_heating_log(y_true[mask], HEATING_LOG_MEAN, HEATING_LOG_STD)
    y_pred_denorm = denormalize_heating_log(y_pred[mask], HEATING_LOG_MEAN, HEATING_LOG_STD)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_denorm, y_pred_denorm, alpha=0.6, color='blue')
    ax.plot([y_true_denorm.min(), y_true_denorm.max()], 
            [y_true_denorm.min(), y_true_denorm.max()], 
            'r--', lw=2, label='Perfekte Vorhersage')
    ax.set_xlabel('Ground Truth (kWh)')
    ax.set_ylabel('Vorhersage (kWh)')
    ax.set_title('Heating Prediction vs Ground Truth')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "heating_scatter.png"), dpi=300, bbox_inches='tight')
    plt.show()


# Dataset preparation
all_folders = sorted([os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR)
                     if os.path.isdir(os.path.join(ROOT_DIR, d))])

HEATING_LOG_MEAN, HEATING_LOG_STD = calculate_heating_log_stats(all_folders)

# PUNKT 4: Stratified split durchführen
if len(all_folders) == 1:
    train_list = val_list = test_list = all_folders
else:
    train_list, val_list, test_list = heating_stratified_split(
        all_folders, test_frac=0.15, val_frac=0.15, random_state=SEED
    )

print(f"Split: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")

# PUNKT 5: Dataset-Analyse durchführen
print("\n" + "="*60)
print("STARTE DATASET-ANALYSE")
print("="*60)

# Gesamtdatensatz-Analyse
overall_stats = evaluate_dataset(all_folders, "Gesamter Datensatz")

# Split-Analyse
evaluate_dataset_splits(train_list, val_list, test_list)

# Zeige Beispielbilder
print("\n--- Beispielbilder aus dem Trainingsdatensatz ---")
show_random_images_with_labels(train_list, n=6)

print("\n" + "="*60)
print("DATASET-ANALYSE ABGESCHLOSSEN")
print("="*60 + "\n")


# FoldersDataset wrapper
class FoldersDataset(MultiTaskDataset):
    def __init__(self, folders_list, *args, **kwargs):
        super().__init__(root_dir=ROOT_DIR, *args, **kwargs)
        self.folders = folders_list
        self.image_paths = []
        for folder in self.folders:
            folder_images = self._find_all_images(folder)
            if not folder_images:
                print(f"Warnung: Keine Bilder in {folder} gefunden!")
            self.image_paths.append(folder_images)

# Create datasets
train_ds = FoldersDataset(train_list, heating_log_mean=HEATING_LOG_MEAN, heating_log_std=HEATING_LOG_STD)
val_ds = FoldersDataset(val_list, heating_log_mean=HEATING_LOG_MEAN, heating_log_std=HEATING_LOG_STD)
test_ds = FoldersDataset(test_list, heating_log_mean=HEATING_LOG_MEAN, heating_log_std=HEATING_LOG_STD)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Dataset sizes: Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Model, Loss, Optimizer
model = DLM_MultiTask_ResNet().to(DEVICE)
criterion_binary = nn.BCEWithLogitsLoss()
criterion_glass = nn.CrossEntropyLoss(ignore_index=IGNORE_GLASS_LABEL)
criterion_heating = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Loss weights (adjust as needed)
LOSS_WEIGHTS = {
    'binary': 1.0,
    'glass': 1.0,
    'heating': 1.0
}


# Training loop
train_losses_epochs = []
val_losses_epochs = {'binary': [], 'glass': [], 'heating': [], 'total': []}
val_metrics_epochs = {'glass_acc': [], 'heating_mae': [], 'heating_rmse': [], 'heating_r2': []}

early_stopping = EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    # PUNKT 6: Separate Task-Losses während Training tracken
    running_losses = {'binary': 0.0, 'glass': 0.0, 'heating': 0.0}
    iters = 0
    
    for imgs, bin_labels, glass_labels, heating_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        imgs = imgs.to(DEVICE)
        bin_labels = bin_labels.to(DEVICE)
        glass_labels = glass_labels.to(DEVICE)
        heating_labels = heating_labels.to(DEVICE)

        optimizer.zero_grad()
        b_logits, g_logits, h_preds = model(imgs)

        # Compute losses
        loss_b = criterion_binary(b_logits, bin_labels)
        loss_g = criterion_glass(g_logits, glass_labels)
        
        mask_heating = ~torch.isnan(heating_labels)
        if mask_heating.any():
            loss_h = criterion_heating(h_preds[mask_heating], heating_labels[mask_heating])
        else:
            loss_h = torch.tensor(0.0, device=DEVICE)

        # Total loss
        total_loss = (LOSS_WEIGHTS['binary'] * loss_b + 
                     LOSS_WEIGHTS['glass'] * loss_g + 
                     LOSS_WEIGHTS['heating'] * loss_h)

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_losses['binary'] += loss_b.item()
        running_losses['glass'] += loss_g.item()
        running_losses['heating'] += loss_h.item()
        iters += 1

    avg_train_loss = running_loss / max(1, iters)
    # PUNKT 7: Ausgabe der separaten Losses
    avg_train_losses = {k: v / max(1, iters) for k, v in running_losses.items()}
    
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")
    print(f"  Binary: {avg_train_losses['binary']:.4f}, Glass: {avg_train_losses['glass']:.4f}, Heating: {avg_train_losses['heating']:.4f}")

    # Validation
    val_stats = evaluate(model, val_loader)
    avg_val_loss = val_stats['losses']['total']
    
    print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}")
    print(f"  Binary Loss: {val_stats['losses']['binary']:.4f}")
    
    # Fix für Glass Accuracy Ausgabe
    glass_acc_str = f"{val_stats['glass_acc']:.3f}" if val_stats['glass_acc'] is not None else 'N/A'
    print(f"  Glass Loss: {val_stats['losses']['glass']:.4f}, Acc: {glass_acc_str}")
    
    print(f"  Heating Loss: {val_stats['losses']['heating']:.4f}")
    if val_stats['heating_mae']:
        print(f"  Heating MAE: {val_stats['heating_mae']:.2f} kWh, RMSE: {val_stats['heating_rmse']:.2f} kWh, R²: {val_stats['heating_r2']:.3f}")
    
    # Store metrics
    train_losses_epochs.append(avg_train_loss)
    for key in val_losses_epochs.keys():
        val_losses_epochs[key].append(val_stats['losses'][key])
    
    val_metrics_epochs['glass_acc'].append(val_stats['glass_acc'] if val_stats['glass_acc'] else np.nan)
    val_metrics_epochs['heating_mae'].append(val_stats['heating_mae'] if val_stats['heating_mae'] else np.nan)
    val_metrics_epochs['heating_rmse'].append(val_stats['heating_rmse'] if val_stats['heating_rmse'] else np.nan)
    val_metrics_epochs['heating_r2'].append(val_stats['heating_r2'] if val_stats['heating_r2'] else np.nan)

    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_multitask.pth")
        print("Saved best model -> best_multitask.pth")

    # Early stopping
    if early_stopping(avg_val_loss, model):
        print(f"Early stopping triggered after {epoch} epochs")
        break

early_stopping.restore_weights(model)
print("Training completed - best weights restored.")


# Test and save results
def test_model_and_write_results(model, test_loader, folders_list, results_dir=RESULTS_DIR):
    ensure_dir(results_dir)
    model.eval()
    
    all_bin_targets, all_bin_preds = [], []
    all_glass_targets, all_glass_preds = [], []
    all_heating_targets, all_heating_preds = [], []
    
    folder_ptr = 0
    inv_glass = {v: k for k, v in GLASS_MAP.items()}

    with torch.no_grad():
        for imgs, bin_labels, glass_labels, heating_labels in test_loader:
            batch_size_actual = imgs.size(0)
            batch_folders = folders_list[folder_ptr: folder_ptr + batch_size_actual]
            folder_ptr += batch_size_actual

            imgs = imgs.to(DEVICE)
            b_logits, g_logits, h_preds = model(imgs)

            # Predictions
            probs = torch.sigmoid(b_logits).cpu().numpy()
            preds_bin = (probs > 0.5).astype(int)
            glass_preds_idx = np.argmax(g_logits.cpu().numpy(), axis=1)

            all_bin_preds.append(preds_bin)
            all_bin_targets.append(bin_labels.cpu().numpy().astype(int))
            all_glass_preds.append(glass_preds_idx)
            all_glass_targets.append(glass_labels.cpu().numpy())
            all_heating_preds.append(h_preds.cpu().numpy())
            all_heating_targets.append(heating_labels.cpu().numpy())

            # PUNKT 8: Write results mit Title-Spalte
            for i, folder in enumerate(batch_folders):
                base_name = os.path.basename(folder.rstrip("/\\"))
                title_val = read_title_from_folder(folder, title_col=TITLE_COLUMN) or base_name
                
                row = {
                    "folder": base_name,
                    "title": title_val  # ← Title hinzugefügt
                }
                
                # Binary
                for j, col in enumerate(BINARY_LABEL_COLUMNS):
                    row[f"{col}_prob"] = float(probs[i, j])
                    row[f"{col}_pred"] = int(preds_bin[i, j])
                
                # Glass
                row["glass_pred"] = inv_glass.get(int(glass_preds_idx[i]), "UNKNOWN")
                
                # Heating
                row["heating_pred"] = float(h_preds[i].cpu().item())

                out_csv_path = os.path.join(results_dir, f"{base_name}_prediction.csv")
                with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)

    # Compute final metrics
    all_bin_targets = np.vstack(all_bin_targets)
    all_bin_preds = np.vstack(all_bin_preds)
    all_glass_targets = np.concatenate(all_glass_targets)
    all_glass_preds = np.concatenate(all_glass_preds)
    all_heating_targets = np.concatenate(all_heating_targets)
    all_heating_preds = np.concatenate(all_heating_preds)

    # Binary metrics
    prf_bin = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average=None, zero_division=0)
    
    # Glass metrics
    mask_valid_glass = all_glass_targets != IGNORE_GLASS_LABEL
    if mask_valid_glass.sum() > 0:
        glass_acc = accuracy_score(all_glass_targets[mask_valid_glass], all_glass_preds[mask_valid_glass])
        prf_glass = precision_recall_fscore_support(
            all_glass_targets[mask_valid_glass], 
            all_glass_preds[mask_valid_glass], 
            average=None, zero_division=0
        )
    else:
        glass_acc = None
        prf_glass = None

    # Heating metrics
    heating_mask = ~np.isnan(all_heating_targets)
    if heating_mask.sum() > 0:
        targets_denorm = denormalize_heating_log(all_heating_targets[heating_mask], HEATING_LOG_MEAN, HEATING_LOG_STD)
        preds_denorm = denormalize_heating_log(all_heating_preds[heating_mask], HEATING_LOG_MEAN, HEATING_LOG_STD)
        heating_mae = mean_absolute_error(targets_denorm, preds_denorm)
        heating_rmse = math.sqrt(mean_squared_error(targets_denorm, preds_denorm))
        heating_r2 = r2_score(targets_denorm, preds_denorm)
    else:
        heating_mae = heating_rmse = heating_r2 = None

    return {
        'binary_prf': prf_bin,
        'glass_acc': glass_acc,
        'glass_prf': prf_glass,
        'heating_mae': heating_mae,
        'heating_rmse': heating_rmse,
        'heating_r2': heating_r2,
        'all_bin_targets': all_bin_targets,
        'all_bin_preds': all_bin_preds,
        'all_glass_targets': all_glass_targets,
        'all_glass_preds': all_glass_preds,
        'all_heating_targets': all_heating_targets,
        'all_heating_preds': all_heating_preds
    }

# Test model
test_stats = test_model_and_write_results(model, test_loader, test_list)

print("\n===== TEST RESULTS =====")
print(f"Glass Accuracy: {test_stats['glass_acc']:.3f}" if test_stats['glass_acc'] else "Glass Accuracy: N/A")
if test_stats['heating_mae']:
    print(f"Heating MAE: {test_stats['heating_mae']:.2f} kWh")
    print(f"Heating RMSE: {test_stats['heating_rmse']:.2f} kWh")
    print(f"Heating R²: {test_stats['heating_r2']:.3f}")

print(f"\nErgebnisse geschrieben nach: {RESULTS_DIR}")

# PUNKT 9: Detaillierte Plots
print("\n===== CREATING PLOTS =====")
ensure_dir(CHARTS_DIR)

# Training Metrics
plot_multitask_training_metrics(
    train_losses_epochs, 
    val_losses_epochs, 
    val_metrics_epochs, 
    save_fig=True
)

# Confusion Matrix (Glass)
plot_confusion_matrix_glass(
    test_stats['all_glass_targets'], 
    test_stats['all_glass_preds'], 
    save_fig=True
)

# F1-Scores (Binary)
plot_binary_f1_scores(test_stats['binary_prf'], save_fig=True)

# Scatter Plot (Heating)
plot_heating_scatter(
    test_stats['all_heating_targets'], 
    test_stats['all_heating_preds'], 
    save_fig=True
)

print("\n[OK] Multi-Task Training abgeschlossen!")