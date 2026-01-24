"""6_1_inference_logp_real_world_prediction.py

Real-world LogP inference for MoleculeNet (Lipo) GAT model.

This script is designed to be resilient:
- Robust paths (works no matter where you run it from)
- Loads optional metadata (*.meta.json) if present
- Supports both scaled-target checkpoints (with inverse transform) and raw-target checkpoints
- Handles PyG from_smiles version differences by trimming/padding node features

Run:
    python 01_property_prediction/6_1_inference_logp_real_world_prediction.py
"""

import os
import json
from pathlib import Path

import torch

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet

from rdkit import Chem
from rdkit import RDLogger

# -----------------------------
# [System Configuration] Suppress Warnings
# -----------------------------
RDLogger.DisableLog("rdApp.*")

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data" / "Lipo"


# -----------------------------
# 1) Model Architecture (must match training)
# -----------------------------
class GAT(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


# -----------------------------
# 2) Checkpoint & metadata selection
# -----------------------------
def load_meta(meta_path: Path):
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def select_artifacts():
    """Prefer scaled checkpoint if present, otherwise fall back to raw."""
    candidates = [
        (SCRIPT_DIR / "best_gat_model_scaled.pth", SCRIPT_DIR / "best_gat_model_scaled.meta.json"),
        (SCRIPT_DIR / "best_gat_model.pth", SCRIPT_DIR / "best_gat_model.meta.json"),
    ]

    for ckpt, meta in candidates:
        if ckpt.exists():
            meta_obj = load_meta(meta) if meta.exists() else None
            return ckpt, meta_obj

    raise FileNotFoundError(
        "No checkpoint found. Expected one of: best_gat_model_scaled.pth, best_gat_model.pth"
    )


def infer_in_dim(meta_obj) -> int:
    """Infer in_dim from metadata, otherwise from local dataset, otherwise fallback to 9."""
    if isinstance(meta_obj, dict) and isinstance(meta_obj.get("in_dim"), int):
        return int(meta_obj["in_dim"])

    # Try dataset
    try:
        ds = MoleculeNet(root=str(DATA_ROOT), name="Lipo")
        return int(ds.num_node_features)
    except Exception:
        return 9


def scaling_info(meta_obj):
    """Return (enabled, scaler_pkl_path, scaler_json_path, mean, scale)."""
    if not isinstance(meta_obj, dict):
        return False, None, None, None, None

    ts = meta_obj.get("target_scaling")
    if not isinstance(ts, dict) or not ts.get("enabled"):
        return False, None, None, None, None

    scaler_pkl = SCRIPT_DIR / str(ts.get("scaler_pkl", "logp_scaler.pkl"))
    scaler_json = SCRIPT_DIR / str(ts.get("scaler_json", "logp_scaler.json"))
    mean = ts.get("mean")
    scale = ts.get("scale")
    return True, scaler_pkl, scaler_json, mean, scale


def load_scaler_params(meta_obj):
    """Load mean/scale either from metadata or from scaler files."""
    enabled, scaler_pkl, scaler_json, mean, scale = scaling_info(meta_obj)
    if not enabled:
        return None

    # Prefer explicit numeric params in metadata
    if isinstance(mean, (int, float)) and isinstance(scale, (int, float)) and float(scale) != 0.0:
        return float(mean), float(scale)

    # Next: load json (portable)
    if scaler_json and scaler_json.exists():
        try:
            obj = json.loads(scaler_json.read_text(encoding="utf-8"))
            m = float(obj["mean"])
            s = float(obj["scale"])
            if s != 0.0:
                return m, s
        except Exception:
            pass

    # Last: load joblib pkl
    if scaler_pkl and scaler_pkl.exists():
        try:
            import joblib

            sc = joblib.load(scaler_pkl)
            m = float(sc.mean_[0])
            s = float(sc.scale_[0])
            if s != 0.0:
                return m, s
        except Exception:
            pass

    print("[Warning] target_scaling enabled but scaler params could not be loaded.")
    return None


# -----------------------------
# 3) SMILES -> Graph conversion
# -----------------------------
def ensure_feature_dim(x: torch.Tensor, in_dim: int) -> torch.Tensor:
    """Trim or zero-pad node features to match model in_dim."""
    if x.dim() != 2:
        return x
    cur = x.size(1)
    if cur == in_dim:
        return x
    if cur > in_dim:
        return x[:, :in_dim]

    # pad
    pad = torch.zeros((x.size(0), in_dim - cur), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def basic_atom_features(atom):
    """A small, stable set of RDKit-derived features (9 dims)."""
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


def smile_to_graph(smile: str, in_dim: int):
    # 1) Try PyG from_smiles
    try:
        try:
            from torch_geometric.utils.smiles import from_smiles
        except Exception:
            from torch_geometric.utils import from_smiles

        data = from_smiles(smile)
        data.x = data.x.to(torch.float)
        data.x = ensure_feature_dim(data.x, in_dim)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        return data
    except Exception as e:
        print(f"[Warning] from_smiles failed ({type(e).__name__}: {e}). Using RDKit fallback.")

    # 2) RDKit fallback
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    x = torch.tensor([basic_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    x = ensure_feature_dim(x, in_dim)

    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    edges += [[j, i] for i, j in edges]

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


# -----------------------------
# 4) Load model
# -----------------------------
def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model():
    ckpt_path, meta_obj = select_artifacts()
    in_dim = infer_in_dim(meta_obj)

    print(f"Checkpoint: {ckpt_path.name}")
    if meta_obj:
        print("Metadata:", json.dumps(meta_obj, indent=2)[:500] + ("..." if len(json.dumps(meta_obj)) > 500 else ""))
    else:
        print("Metadata: (not found)")

    model = GAT(in_dim=in_dim).to(device)
    state = safe_torch_load(ckpt_path)
    model.load_state_dict(state)
    model.eval()

    scaler_params = load_scaler_params(meta_obj)

    return model, in_dim, scaler_params


# -----------------------------
# 5) Inference
# -----------------------------
@torch.no_grad()
def predict_logp(model, smile: str, name: str, in_dim: int, scaler_params=None):
    data = smile_to_graph(smile, in_dim)
    if data is None:
        print(f"[Error] SMILES parsing failed for [{name}]: {smile}")
        return None

    data = data.to(device)
    pred = model(data.x.float(), data.edge_index, data.batch).view(-1)
    pred_item = float(pred.item())

    if scaler_params is None:
        print(f"[{name}] Pred(LogP): {pred_item:.4f}")
        return pred_item

    mean, scale = scaler_params
    pred_raw = pred_item * scale + mean
    print(f"[{name}] Pred(scaled): {pred_item:.4f}  ->  Pred(LogP): {pred_raw:.4f}")
    return pred_raw


@torch.no_grad()
def sanity_check_on_moleculenet(model, in_dim: int, scaler_params=None):
    print("\n[Sanity Check] MoleculeNet(Lipo) sample")
    ds = MoleculeNet(root=str(DATA_ROOT), name="Lipo")
    sample = ds[0].to(device)
    if not hasattr(sample, "batch") or sample.batch is None:
        sample.batch = torch.zeros(sample.num_nodes, dtype=torch.long, device=device)

    x = ensure_feature_dim(sample.x.float(), in_dim)
    pred_scaled = float(model(x, sample.edge_index, sample.batch).view(-1).item())
    y_true = float(sample.y.view(-1).item())

    if scaler_params is None:
        print(f"x_dim(after)={x.size(1)} | model_in_dim={in_dim}")
        print(f"y_true(LogP)={y_true:.4f} | pred(LogP)={pred_scaled:.4f}")
    else:
        mean, scale = scaler_params
        pred_raw = pred_scaled * scale + mean
        print(f"x_dim(after)={x.size(1)} | model_in_dim={in_dim}")
        print(f"y_true(LogP)={y_true:.4f} | pred(scaled)={pred_scaled:.4f} -> pred(LogP)={pred_raw:.4f}")

    print("===============================================\n")


if __name__ == "__main__":
    model, in_dim, scaler_params = load_model()

    sanity_check_on_moleculenet(model, in_dim, scaler_params=scaler_params)

    predict_logp(model, "CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", in_dim, scaler_params=scaler_params)
    predict_logp(model, "CC(=O)NC1=CC=C(C=C1)O", "Tylenol", in_dim, scaler_params=scaler_params)

    print("Done.")
