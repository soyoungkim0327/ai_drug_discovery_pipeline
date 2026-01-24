import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import numpy as np  # Required for array operations
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

# 2. 3D Dataset Generation (InMemoryDataset)
class Lipophilicity3D(InMemoryDataset):
    def __init__(self, root, raw_dataset, transform=None):
        self.raw_dataset = raw_dataset
        super(Lipophilicity3D, self).__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return ['data_3d.pt']

    def process(self):
        data_list = []
        print("Converting 2D SMILES to 3D Conformer structures...")
        for i in tqdm(range(min(500, len(self.raw_dataset)))):
            item = self.raw_dataset[i]
            mol = Chem.MolFromSmiles(item.smiles)
            mol = Chem.AddHs(mol)
            try:
                # 3D Embedding & Optimization
                if AllChem.EmbedMolecule(mol) == -1: continue
                AllChem.MMFFOptimizeMolecule(mol)
                conf = mol.GetConformer()
            except:
                continue

            pos = []
            z = []
            for atom in mol.GetAtoms():
                z.append(atom.GetAtomicNum())
                pos.append(list(conf.GetAtomPosition(atom.GetIdx())))
            
            data = Data(z=torch.tensor(z, dtype=torch.long),
                        pos=torch.tensor(pos, dtype=torch.float),
                        y=item.y)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

from torch_geometric.datasets import MoleculeNet
raw_data = MoleculeNet(root='data/Lipo', name='Lipo')
dataset_3d = Lipophilicity3D(root='data/Lipo_3D', raw_dataset=raw_data)

train_loader = DataLoader(dataset_3d[:400], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset_3d[400:], batch_size=32, shuffle=False)

# 3. Model Architecture Definition (SchNet)
class SchNetModel(torch.nn.Module):
    def __init__(self):
        super(SchNetModel, self).__init__()
        # Continuous-filter convolutional neural network for modeling quantum interactions
        self.model = SchNet(hidden_channels=64, num_filters=64, num_interactions=3, 
                            num_gaussians=50, cutoff=10.0)
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, z, pos, batch):
        h = self.model(z, pos, batch)
        return self.lin(h)

model = SchNetModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Define Training and Evaluation Functions
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.z, data.pos, data.batch)
        loss = F.mse_loss(out.squeeze(), data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(loader):
    model.eval()
    error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.z, data.pos, data.batch)
        error += (out.squeeze() - data.y.squeeze()).abs().sum().item()
    return error / len(loader.dataset)

# ============================================================
# [Main Loop] Training with Early Stopping Strategy
# ============================================================
print("Start 3D SchNet Training (Early Stopping applied)...")

patience = 10           # Patience threshold
patience_counter = 0    
best_mae = float('inf') # Minimize MAE (Initialize with infinity)
max_epochs = 200        # Max epochs

for epoch in range(1, max_epochs + 1):
    loss = train()
    # Evaluate on Test set every epoch
    test_mae = test(test_loader)
    
    # 1. Improvement Detected (MAE decreased)
    if test_mae < best_mae:
        best_mae = test_mae
        patience_counter = 0
        torch.save(model.state_dict(), 'best_schnet_model.pth')
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test MAE: {test_mae:.4f} [New Best Model Saved]')
    
    # 2. No Improvement (Increment patience counter)
    else:
        patience_counter += 1
        if epoch % 5 == 0: # Print log every 5 epochs
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test MAE: {test_mae:.4f} (Waiting... {patience_counter}/{patience})')
        
        # 3. Early Stopping Triggered
        if patience_counter >= patience:
            print(f"\nStop Training (Early Stopping Triggered)")
            print(f"   - Best MAE: {best_mae:.4f}")
            break

print("Task Completed. (Best model saved as 'best_schnet_model.pth')")