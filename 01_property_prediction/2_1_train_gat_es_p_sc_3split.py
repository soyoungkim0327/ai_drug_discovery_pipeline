"""
Standard GAT Training Pipeline (3-Way Split Version)
====================================================
This script implements the Standard GAT architecture with a rigorous
Train/Validation/Test (80/10/10) split strategy.

Key Changes:
- Split: Train (80%) / Val (10%) / Test (10%)
- Early Stopping: Monitors 'Validation MAE' (Not Test MAE) to prevent data leakage.
- Final Evaluation: Evaluates 'Test MAE' only once at the end using the best model.

Configuration:
- Hidden Channels: 64 (Maintained)
- Attention Heads: 4 (Maintained)
- Regularization: None (No Dropout)

Output:
- Artifacts saved with '_3split' suffix to avoid overwriting the winner model.

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-02
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import joblib
import warnings

# Graph Neural Network Imports
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool 
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. Reproducibility & System Configuration
# =============================================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Device Allocation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computational Device: {device}")

# [Path Configuration] Append '_3split' to filenames
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(SCRIPT_DIR, 'logp_gat_scaler_3split.pkl')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_gat_model_3split.pth')

print(f"[*] Artifacts Output Directory: {SCRIPT_DIR}")


# =============================================================================
# 2. Data Acquisition and Preprocessing (3-Way Split)
# =============================================================================
print("\n[Step 1] Loading and Preprocessing Data...")
dataset = MoleculeNet(root='data/Lipo', name='Lipo')

# [Target Standardization]
y_values = dataset.data.y.numpy().reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(y_values) 

joblib.dump(scaler, SCALER_PATH)
print(f"   -> Scaler serialized: '{os.path.basename(SCALER_PATH)}'")

dataset._data.y = torch.from_numpy(scaler.transform(y_values)).float()
print("   -> Dataset Normalization: Z-Score scaling applied.")

# [Data Splitting - Modified for 8:1:1]
# Shuffle dataset first to ensure random distribution
# (IMPORTANT: Without shuffle, the split might be biased by molecular weight etc.)
N = len(dataset)
n_train = int(N * 0.8)
n_val = int(N * 0.1)
n_test = N - n_train - n_val  # Remainder

perm = torch.randperm(N)
dataset = dataset[perm]

train_dataset = dataset[:n_train]
val_dataset = dataset[n_train:n_train+n_val]
test_dataset = dataset[n_train+n_val:]

# [DataLoader Initialization]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)   # For Early Stopping
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # For Final Check

print(f"   -> Data Split: Train({len(train_dataset)}) / Val({len(val_dataset)}) / Test({len(test_dataset)})")


# =============================================================================
# 3. Model Architecture Specification (Standard GAT)
# =============================================================================
class GATModel(torch.nn.Module):
    """
    Standard GAT Model (Unchanged).
    """
    def __init__(self, hidden_channels):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(dataset.num_node_features, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)

model = GATModel(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =============================================================================
# 4. Optimization and Inference Protocols
# =============================================================================
def train():
    """Executes one training epoch."""
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = F.mse_loss(out.squeeze(), data.y.squeeze()) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def evaluate(loader):
    """
    Evaluates model performance on a given loader (Val or Test).
    Returns: Real MAE.
    """
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index, data.batch)
            
            pred_scaled = out.cpu().numpy()
            true_scaled = data.y.cpu().numpy()
            
            pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
            true_real = scaler.inverse_transform(true_scaled.reshape(-1, 1))
            
            error += np.abs(pred_real - true_real).sum()
            
    return error / len(loader.dataset)


# =============================================================================
# 5. Training Campaign with Early Stopping (Monitor Val)
# =============================================================================
print("\n[Step 2] Initiating Training Process (Monitoring Validation Set)...")

PATIENCE = 100
patience_counter = 0
best_val_mae = float('inf') 
MAX_EPOCHS = 1000       

print(f"   -> Config: Hidden=64, Heads=4, No Dropout")
print(f"   -> Max Epochs={MAX_EPOCHS}, Patience={PATIENCE}")
print("-" * 80)
print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Val MAE':<15} | {'Status':<20}")
print("-" * 80)

for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    train_loss = train()
    
    # Evaluate on VALIDATION set (Not Test!)
    val_mae = evaluate(val_loader)
    
    # Early Stopping Logic based on Validation MAE
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0 
        
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"{epoch:<10} | {train_loss:<15.4f} | {val_mae:<15.4f} | {'Saved (Best Val)':<20}")
        
    else:
        patience_counter += 1
        if epoch % 10 == 0 or patience_counter == PATIENCE:
             print(f"{epoch:<10} | {train_loss:<15.4f} | {val_mae:<15.4f} | {'Wait (' + str(patience_counter) + '/' + str(PATIENCE) + ')':<20}")
        
        if patience_counter >= PATIENCE:
            print("-" * 80)
            print(f"[!] Training Terminated (Early Stopping)")
            print(f"    - Best Validation MAE: {best_val_mae:.4f}")
            break

# =============================================================================
# 6. Final Evaluation (The Real Exam)
# =============================================================================
print("\n[Step 3] Final Evaluation on Unseen Test Set...")

# Load the best model found during training
model.load_state_dict(torch.load(MODEL_PATH))

# Evaluate on TEST set (First and last time)
final_test_mae = evaluate(test_loader)

print("=" * 60)
print(f"🏆 FINAL TEST SCORE (Real MAE): {final_test_mae:.4f}")
print("=" * 60)
print(f"   -> This score is statistically valid for papers/interviews.")
print(f"   -> Best Model: '{os.path.basename(MODEL_PATH)}'")
print(f"   -> Location:   {SCRIPT_DIR}")




