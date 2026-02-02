"""
SchNet QM9 Benchmark Training (Energy Prediction U0)
====================================================
This script trains a SchNet model on the QM9 dataset (130k molecules).
Unlike Lipo, QM9 already contains precise DFT-calculated 3D coordinates,
so no RDKit conformation generation is required.

Configuration:
- Dataset: QM9 (130,831 molecules)
- Target: Internal Energy at 0K (U0, Index 7)
- Model: SchNet (Continuous-filter CNN)
- Hidden Channels: 64
- Optimizer: Adam (LR=0.001) with Weight Decay
- Scheduler: ReduceLROnPlateau

Output:
- Saves 'best_schnet_qm9_model.pth' and scaler.

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-03
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import joblib
import json
from datetime import datetime
from tqdm import tqdm

# PyG Imports
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch_geometric.datasets import QM9
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. Reproducibility & Config
# =============================================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computational Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(SCRIPT_DIR, 'qm9_u0_scaler.pkl')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_schnet_qm9_model.pth')
META_PATH = os.path.join(SCRIPT_DIR, 'best_schnet_qm9.meta.json')
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'QM9')


# =============================================================================
# 2. Load QM9 Dataset (No RDKit needed!)
# =============================================================================
print("\n[Step 1] Loading QM9 Dataset (130k Molecules)...")
# PyG automatically downloads & processes if not present
dataset = QM9(root=DATA_PATH)

# QM9 has 19 targets. We select Index 7 (U0: Internal Energy at 0K)
# Index 7 is a standard benchmark for SchNet.
TARGET_IDX = 7 
target_name = "U0 (Internal Energy)"

print(f" -> Dataset Loaded: {len(dataset)} molecules")
print(f" -> Target Property: {target_name} (Index {TARGET_IDX})")

# [Target Standardization]
# Extract the specific target column for scaling
print(" -> Fitting Scaler on Target values...")
all_y = dataset.data.y[:, TARGET_IDX].reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(all_y)
joblib.dump(scaler, SCALER_PATH)

# Normalize the dataset in-memory
# Note: modifying dataset.data.y affects all data in the dataset object
dataset.data.y = torch.from_numpy(scaler.transform(all_y)).float()
print(" -> Z-Score Normalization Applied.")

# [Data Splitting]
# Standard QM9 Split: 110k Train, 10k Val, 10k Test (approx)
# We will use random split for simplicity
N = len(dataset)
n_train = 110000
n_val = 10000
n_test = N - n_train - n_val

gen = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [n_train, n_val, n_test], generator=gen
)

# Batch size increased to 64/128 for speed (QM9 is large)
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f" -> Split: Train({len(train_dataset)}) / Val({len(val_dataset)}) / Test({len(test_dataset)})")

# =============================================================================
# 3. Model & Optimization
# =============================================================================
# [수정된 모델 클래스]
class SchNetRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # SchNet 자체가 이미 [Batch, 1] 크기의 최종 예측값을 내뱉습니다.
        self.model = SchNet(
            hidden_channels=64,
            num_filters=64,
            num_interactions=3,
            num_gaussians=50,
            cutoff=10.0,
        )

    def forward(self, z, pos, batch):
        # SchNet이 다 알아서 계산해줍니다. 그냥 출력하면 됩니다.
        return self.model(z, pos, batch)

print("\n[Step 2] Initializing SchNet Model...")
model = SchNetRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# LR Scheduler: Reduce learning rate when validation loss stops improving
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)

# =============================================================================
# 4. Training Loop
# =============================================================================
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # QM9 data has 'z' (atomic num) and 'pos' (3D coords)
        out = model(data.z, data.pos, data.batch).view(-1)
        target = data.y.view(-1) # Already scaled
        
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate(loader):
    model.eval()
    mae_sum = 0
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.z, data.pos, data.batch).view(-1)
            target = data.y.view(-1)
            
            # Inverse transform for Real MAE (Energy units)
            out_real = scaler.inverse_transform(out.cpu().reshape(-1, 1))
            target_real = scaler.inverse_transform(target.cpu().reshape(-1, 1))
            
            mae_sum += np.abs(out_real - target_real).sum()
            total_samples += data.num_graphs
    return mae_sum / total_samples

# =============================================================================
# 5. Execution
# =============================================================================
print("\n[Step 3] Starting Training (Max 100 Epochs)...")
MAX_EPOCHS = 100 # Reduced for benchmark demo (increase for full convergence)
best_val_mae = float('inf')
patience = 20
patience_counter = 0

print(f"{'Epoch':<5} | {'Train MSE':<12} | {'Val MAE (Real)':<15} | {'LR':<10}")
print("-" * 55)

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss = train()
    val_mae = evaluate(val_loader)
    
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_mae)
    
    print(f"{epoch:<5} | {train_loss:<12.4f} | {val_mae:<15.4f} | {current_lr:.1e}")
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n[!] Early Stopping at Epoch {epoch}")
            break

# =============================================================================
# 6. Final Test
# =============================================================================
print("\n[Step 4] Final Evaluation...")
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    test_mae = evaluate(test_loader)
    print(f"🏆 Final Test MAE (U0 Energy): {test_mae:.4f}")
    
    # Save Meta
    meta = {"model": "SchNet", "dataset": "QM9", "target": "U0", "test_mae": test_mae}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[*] Artifacts saved to: {SCRIPT_DIR}")