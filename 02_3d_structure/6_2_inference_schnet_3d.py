# 6-2.Inference_SchNet.py (FIXED)
import os
import torch
from torch_geometric.nn import SchNet
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")  # Disable RDKit internal warnings

# 1. Environment Configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_schnet_model.pth")


# 2. Model Architecture Wrapper (Must match training configuration)
# Reference: '2.SchNet_3D_Project_ES_P.py'
# Config: SchNet(num_gaussians=50, cutoff=10.0) + Linear(1,1)
class SchNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SchNet(
            hidden_channels=64,
            num_filters=64,
            num_interactions=3,
            num_gaussians=50,
            cutoff=10.0,
        )
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, z, pos, batch):
        h = self.model(z, pos, batch)
        return self.lin(h)


def load_model():
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        print("   -> Please execute training script '2.SchNet_3D_Project_ES_P.py' to generate 'best_schnet_model.pth'.")
        raise SystemExit(1)

    model = SchNetModel().to(device)

    # Load State Dictionary (Handle weights_only parameter for compatibility)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    model.load_state_dict(state)
    model.eval()
    print("3D SchNetModel successfully loaded.")
    return model


@torch.no_grad()
def predict_3d_property(model, smile, name="molecule"):
    print(f"\n[Analysis] Target: {name} | SMILES: {smile}")

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print("[Error] Failed to parse SMILES string.")
        return None

    mol = Chem.AddHs(mol)  # Explicit Hydrogen addition for stable 3D conformation

    # 3D Embedding & Force Field Optimization
    params = AllChem.ETKDGv3()
    ret = AllChem.EmbedMolecule(mol, params)
    if ret == -1:
        print("[Error] 3D Embedding failed (AllChem.EmbedMolecule returned -1).")
        return None

    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        # Proceed with initial coordinates if optimization fails
        pass

    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)

    # Batch vector initialization (Single molecule inference)
    batch = torch.zeros(z.size(0), dtype=torch.long)

    pos = pos.to(device)
    z = z.to(device)
    batch = batch.to(device)

    pred = model(z, pos, batch).item()
    print(f"Predicted 3D Property (y): {pred:.4f}")
    return pred


if __name__ == "__main__":
    model = load_model()

    # Case Study: Aspirin / Tylenol
    predict_3d_property(model, "CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin")
    predict_3d_property(model, "CC(=O)NC1=CC=C(C=C1)O", "Tylenol")