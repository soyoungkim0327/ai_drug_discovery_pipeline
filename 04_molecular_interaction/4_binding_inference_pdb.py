import os
import torch
import urllib.request
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_cluster import radius_graph
from rdkit import Chem
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

# 1. Environment Configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current Device: {device}")

# ---------------------------------------------------------
# [Step 1] Automated PDB Downloader
# ---------------------------------------------------------
def download_pdb(pdb_code, save_dir='./data/PDB'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
    file_path = os.path.join(save_dir, f"{pdb_code}.pdb")
    
    if not os.path.exists(file_path):
        print(f"Downloading: {pdb_code} ...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"Download Complete: {file_path}")
        except Exception as e:
            print(f"Download Failed: {pdb_code} ({e})")
            return None
    else:
        print(f"File already exists: {file_path}")
    
    return file_path

# ---------------------------------------------------------
# [Step 2] PDB to Graph Conversion (Using 3D Coordinates)
# ---------------------------------------------------------
def pdb_to_graph(pdb_path):
    # Load PDB file using RDKit (Sanitize=False to prevent errors on partial structures)
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=True, sanitize=False)
    
    if mol is None:
        print(f"RDKit Parsing Failed: {pdb_path}")
        return None

    # Extract Atom Information
    atoms = []
    coords = []
    
    try:
        conf = mol.GetConformer()
    except:
        return None # Skip if 3D coordinates are missing

    for atom in mol.GetAtoms():
        # Atomic Number (Node Feature)
        atoms.append(atom.GetAtomicNum())
        # 3D Coordinates (x, y, z)
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])

    # Convert to PyTorch Tensor
    x = torch.tensor(atoms, dtype=torch.float).view(-1, 1) # Node Features
    pos = torch.tensor(coords, dtype=torch.float)          # 3D Position Matrix
    
    # Construct Radius Graph (Connect atoms within 5 Angstrom)
    # Generates interaction graph based on spatial proximity
    edge_index = radius_graph(pos, r=5.0, max_num_neighbors=32)
    
    # Create Geometric Data Object
    data = Data(x=x, pos=pos, edge_index=edge_index)
    return data

# ---------------------------------------------------------
# [Step 3] Binding Affinity Prediction Model (GNN)
# ---------------------------------------------------------
class BindingAffinityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # GCN Architecture Definition
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.lin1 = torch.nn.Linear(64, 32)
        self.lin2 = torch.nn.Linear(32, 1) # Final Regressor (Affinity Score)

    def forward(self, x, edge_index, batch):
        # 1. Graph Convolution (Structural Feature Extraction)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        
        # 2. Global Pooling (Readout Layer: Node -> Graph representation)
        x = global_mean_pool(x, batch)
        
        # 3. Prediction Head
        x = self.lin1(x).relu()
        return self.lin2(x)

# ---------------------------------------------------------
# [Step 4] Inference Pipeline Execution
# ---------------------------------------------------------
# 1. Target PDB List (Sample Proteins)
pdb_targets = ['1a2b', '1o86', '3aid'] 
graphs = []
valid_pdbs = []

print("\nStart PDB Data Processing...")
for pdb in pdb_targets:
    path = download_pdb(pdb)
    if path:
        graph = pdb_to_graph(path)
        if graph:
            graphs.append(graph)
            valid_pdbs.append(pdb)

if not graphs:
    print("No valid data to process.")
    exit()

# 2. Model Initialization
model = BindingAffinityModel().to(device)
# Note: Using random weights for structural verification (Load pre-trained weights in production)
model.eval() 

# 3. Execute Inference & Comparison
print("\nAnalyzing Binding Affinity...")
predictions = []

with torch.no_grad():
    for i, data in enumerate(graphs):
        data = data.to(device)
        # Create Batch Vector (Since processing single graphs sequentially)
        batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
        
        # Predict
        score = model(data.x, data.edge_index, batch).item()
        predictions.append(score)
        print(f"[{valid_pdbs[i]}] Predicted Score: {score:.4f}")

# 4. Result Visualization
plt.figure(figsize=(10, 6))
bars = plt.bar(valid_pdbs, predictions, color=['skyblue', 'lightgreen', 'salmon'])

plt.title("Protein Binding Affinity Prediction (AI Analysis)")
plt.xlabel("PDB ID")
plt.ylabel("Predicted Score (Virtual IC50)")
plt.grid(axis='y', linestyle='--')

# Annotate scores on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}', ha='center', va='bottom')

plt.savefig('pdb_comparison.png')
print("\nAnalysis Completed. Results saved to 'pdb_comparison.png' and 'data/PDB' directory.")