# Multi-Class Training Script für "Häuser"-Datensatz
# - Multi-class (Verglasungstyp: 3 Klassen) mit CrossEntropyLoss (ignore_index für fehlende Werte)
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

# Label-Spalten für multi-class-variable "Verglasungstyp"
GLASS_COL = "Verglasungstyp"  # Werte: "Einfachverglasung","Zweifachverglasung","Dreifachverglasung"
GLASS_MAP = {"Einfachverglasung": 0, "Zweifachverglasung": 1, "Dreifachverglasung": 2}
IGNORE_GLASS_LABEL = -100  # wird bei fehlender Angabe verwendet und an CrossEntropyLoss als ignore_index übergeben

# Dataset
class HousesDataset(Dataset):
    def __init__(self, root_dir, glass_col=GLASS_COL, transform=None,
                 image_extensions=('.jpg', '.jpeg', '.png')):
        self.root = root_dir
        # Unterordner, jeweils ein Data-Punkt (Haus)
        self.folders = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
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

        if self.transform:
            img = self.transform(img)

        # Rückgabe: img, glass_label
        return img, glass_label
    

# Model (Backbone + glass head)
class DLM_MultiClass(nn.Module):
    def __init__(self, n_glass_types=len(GLASS_MAP)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.adapt = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(128*4*4, 256)
        self.glass_head = nn.Linear(256, n_glass_types)

    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool(x)
        x = self.relu(self.conv2(x)); x = self.pool(x)
        x = self.relu(self.conv3(x)); x = self.pool(x)
        x = self.adapt(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        g_logits = self.glass_head(x)
        return g_logits

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
    Evaluates the entire dataset and provides statistics about the glass labels.

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
    glass_counts = {glass_type: 0 for glass_type in GLASS_MAP.keys()}
    glass_counts["MISSING"] = 0

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
            glass_counts["MISSING"] += 1
            continue

        # Load CSV
        try:
            df = pd.read_csv(csv_path, sep=";")
        except Exception:
            try:
                df = pd.read_csv(csv_path)  # fallback
            except Exception:
                # Invalid CSV - all values count as missing
                glass_counts["MISSING"] += 1
                continue

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

    # Print results in tabular format
    print("\n===== DATASET EVALUATION =====")
    print(f"Total houses: {len(all_folders)}")

    # Print glass type statistics
    print("\nGLASS TYPES:")
    headers = ["Type", "Count", "Percentage"]

    header_format = "{:<20} {:<10} {:<10}"
    row_format = "{:<20} {:<10} {:<10.1f}"

    print(header_format.format(*headers))
    print("-" * 45)

    total_glass = sum(glass_counts.values())
    for glass_type, count in glass_counts.items():
        percent = (count / total_glass) * 100 if total_glass > 0 else 0
        print(row_format.format(glass_type, count, percent))

    # Create bar chart of glass type distribution
    plt.figure(figsize=(10, 6))
    types = list(glass_counts.keys())
    counts = list(glass_counts.values())
    plt.bar(types, counts, color=['skyblue', 'lightgreen', 'orange', 'red'])
    plt.xlabel('Glass Type')
    plt.ylabel('Count')
    plt.title('Distribution of Glass Types')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)

    # Save chart if CHARTS_DIR exists
    if 'CHARTS_DIR' in globals():
        ensure_dir(CHARTS_DIR)
        plt.savefig(os.path.join(CHARTS_DIR, "glass_type_distribution.png"), dpi=300, bbox_inches='tight')

    plt.show()

    return {
        "total_houses": len(all_folders),
        "glass_counts": glass_counts
    }

dataset_stats = evaluate_dataset(ROOT_DIR)

# Erstelle Datasets und DataLoader
def glass_stratified_split(folders, train_size=0.7, val_size=0.15, test_size=0.15, random_state=SEED):
    """
    Stratifizierte Aufteilung von Ordnern basierend auf Glastypen.
    Stellt sicher, dass alle Glastypen in Train und Val vertreten sind.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split-Anteile müssen sich zu 1.0 summieren"

    # Sammle Labels für alle Ordner
    folder_labels = {}
    for folder in folders:
        csv_path = None
        for f in os.listdir(folder):
            if f.lower().endswith(".csv"):
                csv_path = os.path.join(folder, f)
                break
        glass_type = "MISSING"
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
        folder_labels[folder] = glass_type

    # Stratifizierungslabels
    strat_labels = []
    folders_list = list(folder_labels.keys())
    for folder in folders_list:
        glass_type = folder_labels[folder]
        if glass_type in GLASS_MAP:
            strat_group = f"glass_{glass_type}"
        else:
            strat_group = "glass_MISSING"
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

# Stratifizierte Aufteilung durchführen
all_folders = sorted([os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR)
                     if os.path.isdir(os.path.join(ROOT_DIR, d))])

if len(all_folders) == 1:
    # Nur ein Haus vorhanden: in alle Splits packen
    train_list = all_folders
    val_list = all_folders
    test_list = all_folders
else:
    train_list, val_list, test_list = glass_stratified_split(
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

# Darstellung eines Testbildes
def show_random_train_image_with_title(train_folders, title_col=TITLE_COLUMN):
    ds = FoldersDataset(train_folders)
    idx = random.randint(0, len(ds)-1)

    # unpack 2 values (img, glass)
    sample = ds[idx]
    if len(sample) == 2:
        img_item, glass_label = sample
    else:
        raise RuntimeError(f"Dataset returned {len(sample)} items; erwartet 2 (img, glass). Bitte Klassen neu laden.")

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

    inv_glass_map = {v:k for k,v in GLASS_MAP.items()}
    glass_text = inv_glass_map.get(int(glass_label.cpu().item()), "MISSING") if int(glass_label.cpu().item()) != IGNORE_GLASS_LABEL else "MISSING"

    # Zeige auch Bildpfad und Index-Informationen
    img_filename = os.path.basename(img_path)

    title_str = f"{title_val}\nBild: {img_filename}\nGlass: {glass_text}"
    plt.title(title_str, fontsize=9)
    plt.show()

# Aufruf:
show_random_train_image_with_title(train_list)

# Model, Loss, Optimizer
model = DLM_MultiClass().to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_GLASS_LABEL)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training & Validation
def evaluate(model, loader):
    model.eval()
    all_glass_targets = []
    all_glass_preds = []
    losses = []

    with torch.no_grad():
        for imgs, glass_labels in loader:
            imgs = imgs.to(DEVICE)
            glass_labels = glass_labels.to(DEVICE)

            g_logits = model(imgs)
            
            if (glass_labels != IGNORE_GLASS_LABEL).any():
                loss_g = criterion(g_logits, glass_labels)
                losses.append(loss_g.item())

            # glass preds
            glass_targets_np = glass_labels.cpu().numpy()
            glass_pred_np = np.argmax(g_logits.cpu().numpy(), axis=1)
            all_glass_targets.append(glass_targets_np)
            all_glass_preds.append(glass_pred_np)

    # concat
    all_glass_targets = np.concatenate(all_glass_targets) if all_glass_targets else np.array([], dtype=int)
    all_glass_preds = np.concatenate(all_glass_preds) if all_glass_preds else np.array([], dtype=int)

    # Glass metrics (ignore IGNORE index)
    mask_valid = all_glass_targets != IGNORE_GLASS_LABEL
    if mask_valid.sum() > 0:
        glass_acc = accuracy_score(all_glass_targets[mask_valid], all_glass_preds[mask_valid])
        prf_glass = precision_recall_fscore_support(all_glass_targets[mask_valid], all_glass_preds[mask_valid], average=None, zero_division=0)
    else:
        glass_acc = None
        prf_glass = None

    avg_loss_g = np.mean(losses) if losses else 0.0

    return {
        "glass_prf": prf_glass,
        "glass_acc": glass_acc,
        "loss_g": avg_loss_g,
        "all_glass_targets": all_glass_targets,
        "all_glass_preds": all_glass_preds
    }

# Training loop mit Early Stopping
train_losses_epochs = []
val_loss_g_epochs = []
val_glass_acc_epochs = []

# Early Stopping initialisieren
early_stopping = EarlyStopping(patience=5, min_delta=0.001, restore_best_weights=True)

# Initialisiere best_val_loss mit einem sehr hohen Wert
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    iters = 0
    for imgs, glass_labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        imgs = imgs.to(DEVICE)
        glass_labels = glass_labels.to(DEVICE)

        optimizer.zero_grad()
        g_logits = model(imgs)

        if (glass_labels != IGNORE_GLASS_LABEL).any():
            loss = criterion(g_logits, glass_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iters += 1

    avg_train_loss = running_loss / max(1, iters)
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")

    # Validation
    val_stats = evaluate(model, val_loader)
    avg_val_loss = val_stats['loss_g']
    print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}")

    # Metriken speichern für Plot
    train_losses_epochs.append(avg_train_loss)
    val_loss_g_epochs.append(avg_val_loss)
    val_glass_acc_epochs.append(val_stats['glass_acc'] if val_stats['glass_acc'] is not None else np.nan)

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

    # Modell-Checkpointing (bestes Modell basierend auf Validation-Loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_glassnet.pth")
        print("Saved best model -> best_glassnet.pth")

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
    all_glass_targets = []
    all_glass_preds = []
    losses = []

    folder_ptr = 0
    inv_glass = {v:k for k,v in GLASS_MAP.items()}

    with torch.no_grad():
        for imgs, glass_labels in test_loader:
            batch_size_actual = imgs.size(0)
            batch_folders = folders[folder_ptr: folder_ptr + batch_size_actual]
            folder_ptr += batch_size_actual

            imgs = imgs.to(DEVICE)
            glass_labels = glass_labels.to(DEVICE)

            g_logits = model(imgs)

            # Calculate losses
            if (glass_labels != IGNORE_GLASS_LABEL).any():
                loss_g = criterion(g_logits, glass_labels).item()
                losses.append(loss_g)

            # Predictions
            glass_preds_idx = np.argmax(g_logits.cpu().numpy(), axis=1)
            all_glass_preds.append(glass_preds_idx)
            all_glass_targets.append(glass_labels.cpu().numpy())

            # Write results for each element in the batch
            for i, folder in enumerate(batch_folders):
                base_name = os.path.basename(folder.rstrip("/\\"))
                title_val = read_title_from_folder(folder, title_col=TITLE_COLUMN) or base_name

                row = {
                    "folder": base_name,
                    "title": title_val
                }

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

                # write CSV (one file per object)
                out_csv_path = os.path.join(results_dir, f"{base_name}_prediction.csv")
                with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerow(row)

    # concat arrays for aggregated metrics
    all_glass_targets = np.concatenate(all_glass_targets) if all_glass_targets else np.array([], dtype=int)
    all_glass_preds   = np.concatenate(all_glass_preds)   if all_glass_preds   else np.array([], dtype=int)

    # Glass metrics (ignore IGNORE index)
    mask_valid = all_glass_targets != IGNORE_GLASS_LABEL
    if mask_valid.sum() > 0:
        glass_acc = accuracy_score(all_glass_targets[mask_valid], all_glass_preds[mask_valid])
        prf_glass = precision_recall_fscore_support(all_glass_targets[mask_valid], all_glass_preds[mask_valid], average=None, zero_division=0)
    else:
        glass_acc = None
        prf_glass = None

    avg_loss_g = np.mean(losses) if losses else 0.0

    return {
        "glass_prf": prf_glass,
        "glass_acc": glass_acc,
        "loss_g": avg_loss_g,
        "all_glass_targets": all_glass_targets,
        "all_glass_preds": all_glass_preds
    }

# Performance Metriken
ensure_dir(CHARTS_DIR)

def plot_glass_training_metrics(train_losses_epochs, val_loss_g_epochs, val_glass_acc_epochs, glass_prf, test_stats, save_fig=True):
    """
    Plots training loss curves, glass metrics per class, and confusion matrix for glass classification.
    """
    # 1. Training Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(train_losses_epochs)+1), train_losses_epochs, label='Train Loss', color='blue', marker='o')
    plt.plot(np.arange(1, len(val_loss_g_epochs)+1), val_loss_g_epochs, label='Val Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Glass Classification Loss per Epoch')
    plt.legend()
    plt.grid(alpha=0.2)
    if save_fig:
        plt.savefig(os.path.join(CHARTS_DIR, "glass_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Glass Accuracy Plot
    plt.figure(figsize=(8, 5))
    # Filter out NaN values for plotting
    epochs = np.arange(1, len(val_glass_acc_epochs)+1)
    valid_mask = ~np.isnan(val_glass_acc_epochs)
    if np.any(valid_mask):
        plt.plot(epochs[valid_mask], np.array(val_glass_acc_epochs)[valid_mask], color='orange', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Glass Classification Accuracy per Epoch')
        plt.ylim([0, 1.05])
        plt.grid(alpha=0.2)
        if save_fig:
            plt.savefig(os.path.join(CHARTS_DIR, "glass_accuracy_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Glass Metrics per Class (Precision, Recall, F1)
    if glass_prf is not None and len(glass_prf) == 4:
        prec, rec, f1, support = glass_prf
        
        # Get the actual available glass classes
        available_classes = len(prec)
        
        if available_classes > 0:
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

            plt.figure(figsize=(10, 6))
            plt.bar(x - width, prec, width, label='Precision')
            plt.bar(x, rec, width, label='Recall')
            plt.bar(x + width, f1, width, label='F1')
            plt.xlabel('Glass Types')
            plt.ylabel('Score')
            plt.title('Glass Classification Metrics per Class')
            plt.xticks(x, glass_labels, rotation=45, ha='right')
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            if save_fig:
                plt.savefig(os.path.join(CHARTS_DIR, "glass_metrics_per_class.png"), dpi=300, bbox_inches='tight')
            plt.show()

    # 4. Confusion Matrix for glass classification
    if test_stats is not None and 'all_glass_targets' in test_stats and 'all_glass_preds' in test_stats:
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

# Nach Training und Test aufrufen:
# Testen des Modells und Berechnung der Test-Metriken
test_stats = test_model_and_write_results(model, test_loader)

print("Test Loss G:{:.4f}".format(test_stats['loss_g']))
if test_stats['glass_acc'] is not None:
    print(f"Glass Accuracy: {test_stats['glass_acc']:.3f}")

print("Ergebnisse geschrieben nach:", RESULTS_DIR)

# Jetzt die Plots erzeugen
plot_glass_training_metrics(
    train_losses_epochs=train_losses_epochs,
    val_loss_g_epochs=val_loss_g_epochs,
    val_glass_acc_epochs=val_glass_acc_epochs,
    glass_prf=test_stats['glass_prf'],
    test_stats=test_stats,
    save_fig=True
)