"""
Standard GCN Training Pipeline
==============================
This script implements a Graph Convolutional Network (GCN) for molecular property prediction.
It serves as a baseline model to compare against GAT architectures.

Configuration:
- Architecture: 3-Layer GCN (Hidden Channels: 64)
- Optimization: Adam (LR=0.005)
- Stability: Patience 100 with Early Stopping
- Monitoring: Tracks both Training Loss and Real-world MAE.

Output:
- Artifacts (Model/Scaler) are saved in the SAME directory as this script.
- Model saved as: 'best_gnn_model.pth'

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
# Ensure deterministic behavior for scientific reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

# Suppress system-specific warnings
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Device Allocation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computational Device: {device}")

# [Path Configuration] Force artifacts to save in the script's directory
# This ensures files are saved exactly where this .py file resides.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(SCRIPT_DIR, 'logp_gnn_scaler.pkl')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_gnn_model.pth')

print(f"[*] Artifacts Output Directory: {SCRIPT_DIR}")


# =============================================================================
# 2. Data Acquisition and Preprocessing
# =============================================================================
print("\n[Step 1] Loading and Preprocessing Data...")
dataset = MoleculeNet(root='data/Lipo', name='Lipo')

# [Target Standardization]
# Use ._data to avoid PyG warnings as per original code
y_values = dataset._data.y.numpy().reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(y_values) 

# Serialize Scaler using Absolute Path
joblib.dump(scaler, SCALER_PATH)
print(f"   -> Scaler serialized: '{os.path.basename(SCALER_PATH)}'")

# Apply Transformation
dataset._data.y = torch.from_numpy(scaler.transform(y_values)).float()
print("   -> Dataset Normalization: Z-Score scaling applied.")

# [Data Splitting]
train_dataset = dataset[:3300]
test_dataset = dataset[3300:]

# [DataLoader Initialization]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"   -> Train Batches: {len(train_loader)}, Test Batches: {len(test_loader)}")


# =============================================================================
# 3. Model Architecture Specification (GCN)
# =============================================================================
class GCNModel(torch.nn.Module):
    """
    Standard GCN Model.
    
    Architecture:
        - 3x GCNConv Layers
        - Hidden Channels: 64
        - Global Mean Pooling
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
    """
    Executes one training epoch via backpropagation.
    Returns: Average Training Loss (MSE)
    """
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

def test(loader):
    """
    Evaluates model performance on the test set.
    Returns: Real Mean Absolute Error (Inverse Transformed)
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
# 5. Training Campaign with Early Stopping
# =============================================================================
print("\n[Step 2] Initiating Training Process (GCN Model)...")

# [Hyperparameters]
PATIENCE = 100          # Increased patience
patience_counter = 0    
best_mae = float('inf') 
MAX_EPOCHS = 1000       

print(f"   -> Config: Hidden=64, Optimizer=Adam(lr=0.005)")
print(f"   -> Max Epochs={MAX_EPOCHS}, Patience={PATIENCE}")
print("-" * 80)
print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Real MAE':<15} | {'Status':<20}")
print("-" * 80)

try:
    for epoch in range(1, MAX_EPOCHS + 1):
        # Execute Training & Validation
        train_loss = train()
        test_mae = test(test_loader)
        
        # [Check Point 1] Performance Improvement
        if test_mae < best_mae:
            best_mae = test_mae
            patience_counter = 0 
            
            # Save Model Artifact (Using Absolute Path)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"{epoch:<10} | {train_loss:<15.4f} | {test_mae:<15.4f} | {'Saved (*)':<20}")
            
        # [Check Point 2] Stagnation
        else:
            patience_counter += 1
            if epoch % 10 == 0 or patience_counter == PATIENCE:
                 print(f"{epoch:<10} | {train_loss:<15.4f} | {test_mae:<15.4f} | {'Wait (' + str(patience_counter) + '/' + str(PATIENCE) + ')':<20}")
            
            # [Check Point 3] Early Stopping Trigger
            if patience_counter >= PATIENCE:
                print("-" * 80)
                print(f"[!] Training Terminated (Early Stopping)")
                print(f"    - Best Real MAE: {best_mae:.4f}")
                break
                
except Exception as e:
    print(f"\n[CRITICAL ERROR] Training crashed: {e}")
    sys.exit(1)

print("\n[Step 3] Pipeline Completed.")
print(f"   -> Best Model: '{os.path.basename(MODEL_PATH)}'")
print(f"   -> Location:   {SCRIPT_DIR}")