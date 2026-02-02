import os
import joblib
import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet

from rdkit import Chem
from rdkit import RDLogger

# -----------------------------
# [System Configuration] Suppress Warnings and Initialize Logging
# -----------------------------
RDLogger.DisableLog("rdApp.*")  # Disable RDKit internal warnings for cleaner output

print("SCRIPT START")

# -----------------------------
# 1. Environment Setup
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_gat_model.pth")
scaler_path = os.path.join(current_dir, "logp_scaler.pkl")

# Input Feature Dimension (Fixed to 9 based on GATConv configuration)
IN_DIM = 9

# Target Value Range (Derived from Lipo dataset statistics)
Y_MIN, Y_MAX = -1.5, 4.5

# -----------------------------
# 2. Model Architecture Definition (Consistent with Training Phase)
# -----------------------------
class GAT(torch.nn.Module):
    def __init__(self, in_dim=IN_DIM):
        super().__init__()
        self.conv1 = GATConv(in_dim, 64, heads=4, concat=False)
        self.conv2 = GATConv(64, 64, heads=4, concat=False)
        self.conv3 = GATConv(64, 64, heads=4, concat=False)
        self.lin = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# -----------------------------
# 3. Data Preprocessing: SMILES to Graph Conversion
#   - Priority 1: PyG from_smiles utility
#   - Fallback: Manual RDKit feature extraction (if input dimension mismatch occurs)
# -----------------------------
def get_atom_features_9(atom):
    # Define 9 atomic features for fallback compatibility
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        atom.GetNumRadicalElectrons(),
        atom.GetImplicitValence(),
        int(atom.IsInRing()),
    ]

def smile_to_graph_rdkit9(smile: str):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    x = torch.tensor([get_atom_features_9(a) for a in mol.GetAtoms()], dtype=torch.float)

    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    edges += [[j, i] for i, j in edges]

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def smile_to_graph(smile: str):
    # 1) Attempt using PyG from_smiles
    try:
        # Handle import path variations across PyG versions
        try:
            from torch_geometric.utils.smiles import from_smiles
        except Exception:
            from torch_geometric.utils import from_smiles

        data = from_smiles(smile)
        data.x = data.x.to(torch.float)

        # Batch initialization for single-instance inference
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        x_dim = data.x.size(1)
        print(f"PyG from_smiles extracted x_dim = {x_dim}")

        # Check for dimension mismatch against model requirements
        if x_dim != IN_DIM:
            print(
                f"[Warning] Dimension Mismatch: extracted x_dim({x_dim}) != IN_DIM({IN_DIM}). "
                "Reverting to fallback (RDKit 9-feature) method."
            )
            return smile_to_graph_rdkit9(smile)

        return data

    except Exception as e:
        print(f"[Warning] from_smiles failed ({type(e).__name__}: {e}). Reverting to fallback method.")
        return smile_to_graph_rdkit9(smile)


# -----------------------------
# 4. Model & Scaler Loading
# -----------------------------
def load_model_and_scaler():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = GAT(in_dim=IN_DIM).to(device)

    # Secure model loading (Support 'weights_only' parameter if available)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    model.load_state_dict(state)
    model.eval()

    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully:", scaler_path)
        except Exception as e:
            print(f"[Warning] Scaler load failed ({type(e).__name__}: {e}). Proceeding without scaler.")
            scaler = None
    else:
        print("[Warning] Scaler file not found. Outputting raw predictions.")

    return model, scaler


# -----------------------------
# 5. Inference Execution
# -----------------------------
@torch.no_grad()
def predict_raw(model, smile: str, name: str):
    data = smile_to_graph(smile)
    if data is None:
        print(f"[Error] SMILES parsing failed for [{name}]: {smile}")
        return None

    data = data.to(device)
    pred = model(data.x.float(), data.edge_index, data.batch).item()
    print(f"[{name}] Raw Prediction: {pred:.4f}")
    return pred


# -----------------------------
# 6. Sanity Check (Model Validation on Benchmark Data)
# -----------------------------
@torch.no_grad()
def sanity_check_on_moleculenet(model):
    print("\n[Sanity Check] Validating on MoleculeNet (Lipo) Sample")
    ds = MoleculeNet(root="data/Lipo", name="Lipo")

    sample = ds[0].to(device)
    # Ensure batch attribute exists for single-graph processing
    if not hasattr(sample, "batch") or sample.batch is None:
        sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long, device=device)

    x_dim = sample.x.size(1)
    print(f"Sample x_dim = {x_dim} | Model IN_DIM = {IN_DIM}")
    print(f"Ground Truth y = {float(sample.y):.4f}")
    print(f"SMILES = {getattr(sample, 'smiles', '(no smiles field)')}")

    if x_dim != IN_DIM:
        print("[Warning] Feature dimension mismatch between MoleculeNet sample and Model.")
        print("   -> Retraining required with IN_DIM set to dataset.num_node_features.")

    pred = model(sample.x.float(), sample.edge_index, sample.batch).item()
    print(f"Prediction on MoleculeNet sample = {pred:.4f}")

    # Inspect target value distribution (First 50 samples)
    ys = []
    for i in range(min(50, len(ds))):
        ys.append(float(ds[i].y))
    print(f"Target Distribution (First 50): min={min(ys):.3f}, max={max(ys):.3f}")
    print("===============================================\n")


if __name__ == "__main__":
    print("[Real-World Prediction + Sanity Check] Pipeline Initialized\n")

    model, scaler = load_model_and_scaler()

    # 1) Validate Model Integrity (Critical Step)
    sanity_check_on_moleculenet(model)

    # 2) Execute Inference on Target Molecules
    predict_raw(model, "CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin")
    predict_raw(model, "CC(=O)NC1=CC=C(C=C1)O", "Tylenol")

    print("\nProcess Completed.")