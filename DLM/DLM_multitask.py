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


# Helper functions
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

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


# Dataset preparation
all_folders = sorted([os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR)
                     if os.path.isdir(os.path.join(ROOT_DIR, d))])

HEATING_LOG_MEAN, HEATING_LOG_STD = calculate_heating_log_stats(all_folders)

# Stratified split (simplified)
if len(all_folders) == 1:
    train_list = val_list = test_list = all_folders
else:
    # Simple random split (you can use your existing stratified split functions)
    train_list, test_list = train_test_split(all_folders, test_size=0.15, random_state=SEED)
    train_list, val_list = train_test_split(train_list, test_size=0.176, random_state=SEED)  # 0.176*0.85 ≈ 0.15

print(f"Split: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")

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

            # Heating loss
            mask_heating = ~torch.isnan(heating_labels)
            if mask_heating.any():
                loss_h = criterion_heating(h_preds[mask_heating], heating_labels[mask_heating])
                losses['heating'].append(loss_h.item())

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


# Training loop
train_losses_epochs = []
val_losses_epochs = {'binary': [], 'glass': [], 'heating': [], 'total': []}
val_metrics_epochs = {'glass_acc': [], 'heating_mae': [], 'heating_rmse': [], 'heating_r2': []}

early_stopping = EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)
best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
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
        iters += 1

    avg_train_loss = running_loss / max(1, iters)
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}")

    # Validation
    val_stats = evaluate(model, val_loader)
    avg_val_loss = val_stats['losses']['total']
    
    print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}")
    print(f"  Binary Loss: {val_stats['losses']['binary']:.4f}")
    print(f"  Glass Loss: {val_stats['losses']['glass']:.4f}, Acc: {val_stats['glass_acc']:.3f if val_stats['glass_acc'] else 'N/A'}")
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

            # Write results
            for i, folder in enumerate(batch_folders):
                base_name = os.path.basename(folder.rstrip("/\\"))
                row = {"folder": base_name}
                
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

# Optional: Plotting (you can add comprehensive plots here)
ensure_dir(CHARTS_DIR)

plt.figure(figsize=(10, 6))
plt.plot(train_losses_epochs, label='Train Loss')
plt.plot(val_losses_epochs['total'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Multi-Task Training Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(CHARTS_DIR, "multitask_loss.png"), dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Multi-Task Training abgeschlossen!")