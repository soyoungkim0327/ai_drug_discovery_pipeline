"""
Standard GCN Training Pipeline (3-Way Split Version)
====================================================
This script implements the Standard GCN architecture with a rigorous
Train/Validation/Test (80/10/10) split strategy.

Key Changes:
- Split: Train (80%) / Val (10%) / Test (10%)
- Early Stopping: Monitors 'Validation MAE' to prevent data leakage.
- Final Evaluation: Evaluates 'Test MAE' only once at the end.

Configuration:
- Architecture: 3-Layer GCN (Hidden Channels: 64)
- Optimization: Adam (LR=0.005)
- Stability: Patience 100 with Early Stopping

Output:
- Artifacts saved with '_3split' suffix.

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-02
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import joblib
import sys
import warnings

# Graph Neural Network Imports
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool 
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

# [Path Configuration] Force artifacts to save in the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(SCRIPT_DIR, 'logp_gnn_scaler_3split.pkl')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_gnn_model_3split.pth')

print(f"[*] Artifacts Output Directory: {SCRIPT_DIR}")


# =============================================================================
# 2. Data Acquisition and Preprocessing (3-Way Split)
# =============================================================================
print("\n[Step 1] Loading and Preprocessing Data...")
dataset = MoleculeNet(root='data/Lipo', name='Lipo')

# [Target Standardization]
y_values = dataset._data.y.numpy().reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(y_values) 

# Serialize Scaler using Absolute Path
joblib.dump(scaler, SCALER_PATH)
print(f"   -> Scaler serialized: '{os.path.basename(SCALER_PATH)}'")

# Apply Transformation
dataset._data.y = torch.from_numpy(scaler.transform(y_values)).float()
print("   -> Dataset Normalization: Z-Score scaling applied.")

# [Data Splitting: 80% Train / 10% Val / 10% Test]
N = len(dataset)
n_train = int(N * 0.8)
n_val = int(N * 0.1)
n_test = N - n_train - n_val

# Shuffle dataset to ensure random distribution
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
# 3. Model Architecture Specification (GCN)
# =============================================================================
class GCNModel(torch.nn.Module):
    """
    Standard GCN Model (Unchanged).
    """
    def __init__(self, hidden_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)

# [Model Initialization]
print("[Step 4] Initializing Model...")
model = GCNModel(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


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
    Evaluates model performance on a specific loader (Val or Test).
    Returns: Real MAE (Inverse Transformed).
    """
    model.eval()
    error = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x.float(), data.edge_index, data.batch)
            
            # [Inverse Transform for Real Metrics]
            pred_scaled = out.cpu().detach().numpy()
            true_scaled = data.y.cpu().detach().numpy()
            
            pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
            true_real = scaler.inverse_transform(true_scaled.reshape(-1, 1))
            
            error += np.abs(pred_real - true_real).sum()
            
    return error / len(loader.dataset)


# =============================================================================
# 5. Training Campaign with Early Stopping (Monitor Val)
# =============================================================================
print("\n[Step 2] Initiating Training Process (GCN Model - 3 Split)...")

# [Hyperparameters]
PATIENCE = 100          
patience_counter = 0    
best_val_mae = float('inf')  # Monitor Validation Score
MAX_EPOCHS = 1000       

print(f"   -> Config: Hidden=64, Optimizer=Adam(lr=0.005)")
print(f"   -> Max Epochs={MAX_EPOCHS}, Patience={PATIENCE}")
print("-" * 80)
print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Val MAE':<15} | {'Status':<20}")
print("-" * 80)

try:
    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        train_loss = train()
        
        # Evaluate on Validation Set
        val_mae = evaluate(val_loader)
        
        # [Check Point 1] Performance Improvement (on Val)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0 
            
            # Save Model Artifact
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"{epoch:<10} | {train_loss:<15.4f} | {val_mae:<15.4f} | {'Saved (Best Val)':<20}")
            
        # [Check Point 2] Stagnation
        else:
            patience_counter += 1
            if epoch % 10 == 0 or patience_counter == PATIENCE:
                 print(f"{epoch:<10} | {train_loss:<15.4f} | {val_mae:<15.4f} | {'Wait (' + str(patience_counter) + '/' + str(PATIENCE) + ')':<20}")
            
            # [Check Point 3] Early Stopping Trigger
            if patience_counter >= PATIENCE:
                print("-" * 80)
                print(f"[!] Training Terminated (Early Stopping)")
                print(f"    - Best Validation MAE: {best_val_mae:.4f}")
                break
                
except Exception as e:
    print(f"\n[CRITICAL ERROR] Training crashed: {e}")
    sys.exit(1)

# =============================================================================
# 6. Final Test Evaluation (Once at the end)
# =============================================================================
print("\n[Step 3] Final Evaluation on Unseen Test Set...")

# Load the best model saved during validation
model.load_state_dict(torch.load(MODEL_PATH))

# Evaluate on Test Set
final_test_mae = evaluate(test_loader)

print("=" * 60)
print(f"🏆 FINAL TEST SCORE (Real MAE): {final_test_mae:.4f}")
print("=" * 60)
print(f"   -> Best Model: '{os.path.basename(MODEL_PATH)}'")
print(f"   -> Location:   {SCRIPT_DIR}")