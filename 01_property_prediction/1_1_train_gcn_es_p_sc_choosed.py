import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import os
import torch
import numpy as np
import random  # Added for reproducibility

# --- [Reproducibility Configuration] ---
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False
# --------------------

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ...

# [System Config] Prevent OpenMP conflict error on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# [Device Config] Check CUDA availability and assign device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Current Device: {device}")


# 1. Environment Configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in use: {device}")

# 2. Data Preparation
dataset = MoleculeNet(root='data/Lipo', name='Lipo')
train_dataset = dataset[:3300]
test_dataset = dataset[3300:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

from sklearn.preprocessing import StandardScaler
import joblib # Required for saving Scaler

# 2. Data Preparation and Scaler Application
dataset = MoleculeNet(root='data/Lipo', name='Lipo')

# Extract Y values (LogP) and fit Scaler
y_values = dataset.data.y.numpy().reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(y_values)

# Save Scaler to file for inference
joblib.dump(scaler, 'logp_scaler.pkl')
print("Scaler saved: 'logp_scaler.pkl'")

# Note: Model will be trained using transformed y values in subsequent steps



# 3. Model Architecture Definition (GCN)
class GCNModel(torch.nn.Module):
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

model = GCNModel(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Define Training and Evaluation Functions
def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = F.mse_loss(out.squeeze(), data.y.squeeze()) 
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch)
        error += (out.squeeze() - data.y.squeeze()).abs().sum().item()
    return error / len(loader.dataset)

# ============================================================
# [Main Loop] Training Pipeline with Early Stopping Mechanism
# ============================================================
print("Start Training (Early Stopping applied)...")

# Hyperparameters and Early Stopping Settings
patience = 20           # Patience threshold
patience_counter = 0    # Current patience counter
best_mae = float('inf') # Initialize best MAE
max_epochs = 1000       # Max epochs

for epoch in range(1, max_epochs + 1):
    train() # Execute Training Step
    
    # Evaluate on Test set every epoch
    test_mae = test(test_loader)
    
    # 1. Improvement Detected (MAE decreased)
    if test_mae < best_mae:
        best_mae = test_mae
        patience_counter = 0 # Reset counter
        
        # Save best model weights
        torch.save(model.state_dict(), 'best_gcn_model.pth')
        print(f'Epoch: {epoch:03d}, MAE: {test_mae:.4f} [New Best Model Saved]')
        
    # 2. No Improvement (Increment patience counter)
    else:
        patience_counter += 1
        if epoch % 10 == 0: # Print log every 10 epochs
             print(f'Epoch: {epoch:03d}, MAE: {test_mae:.4f} (Waiting... {patience_counter}/{patience})')
        
        # 3. Early Stopping Triggered
        if patience_counter >= patience:
            print(f"\nStop Training (Early Stopping Triggered)")
            print(f"   - No improvement for {patience} epochs.")
            print(f"   - Best MAE: {best_mae:.4f}")
            break

print("Task Completed. (Best model saved as 'best_gcn_model.pth')")