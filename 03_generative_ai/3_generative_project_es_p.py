import os
import torch
import numpy as np # Added for reproducibility (Explicit import)
from torch_geometric.datasets import MoleculeNet
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, VGAE
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

# 1. Environment Configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current Device: {device}")

# 2. Data Preparation
# [Pre-processing] Explicit float conversion applied for stability
dataset = MoleculeNet(root='data/Lipo', name='Lipo')
data = dataset[0]  
data.x = data.x.float() # Critical: Type casting to float32

# Link Split for Generative Task
transform = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                            split_labels=True, add_negative_train_samples=False)
train_data, val_data, test_data = transform(data)

train_data = train_data.to(device)
test_data = test_data.to(device)

# 3. Model Architecture Definition (VGAE)
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

out_channels = 16
model = VGAE(VariationalGCNEncoder(dataset.num_features, out_channels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Define Training and Evaluation Functions
def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def test(data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

# ============================================================
# [Main Loop] Training with Early Stopping (Metric: AUC)
# ============================================================
print("Start VGAE Training (Early Stopping applied)...")

patience = 20
patience_counter = 0
best_auc = 0.0          # Monitor AUC (Maximize)
max_epochs = 1000

for epoch in range(1, max_epochs + 1):
    loss = train()
    auc, ap = test(test_data)
    
    # 1. Improvement Detected (New Best AUC)
    if auc > best_auc:
        best_auc = auc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_vgae_model.pth')
        # Log only on improvement
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f} [New Best Model Saved]')
        
    # 2. No Improvement (Increment Patience)
    else:
        patience_counter += 1
        # Print log every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f} (Waiting... {patience_counter}/{patience})')
            
        # 3. Early Stopping Triggered
        if patience_counter >= patience:
            print(f"\nStop Training (Early Stopping Triggered)")
            print(f"   - Best AUC: {best_auc:.4f}")
            break

print("Training Completed. ('best_vgae_model.pth' saved)")