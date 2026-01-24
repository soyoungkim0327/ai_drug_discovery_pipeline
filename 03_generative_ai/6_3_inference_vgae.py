# 6-3.Inference_VGAE.py (FIXED: load existing pth + float-cast before NormalizeFeatures)
import os
from pathlib import Path
import torch

from torch_geometric.datasets import MoleculeNet
from torch_geometric.transforms import NormalizeFeatures, RandomLinkSplit
from torch_geometric.nn import GCNConv, VGAE

# 1. Environment Configuration
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "best_vgae_model.pth")

project_root = Path(current_dir).parent
data_root = project_root / "data" / "Lipo"


# 2. Preprocessing Utility: Type Casting (Int64 -> Float32)
# Handles compatibility issues with NormalizeFeatures transform
class CastXToFloat:
    def __call__(self, data):
        if hasattr(data, "x") and data.x is not None:
            data.x = data.x.to(torch.float)
        return data


# 3. Model Architecture Definition (Variational Graph Autoencoder)
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def _safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_dims_from_state_dict(state):
    """
    Infer input/output dimensions from model state dictionary.
    - conv1.lin.weight shape: [2*out, in]
    - conv_mu.lin.weight shape: [out, 2*out]
    """
    in_ch = None
    out_ch = None

    # Infer in_channels from conv1
    for k, v in state.items():
        if k.endswith("conv1.lin.weight") or k.endswith("encoder.conv1.lin.weight"):
            # Shape: [2*out, in]
            in_ch = int(v.shape[1])
            break

    # Infer out_channels from conv_mu
    for k, v in state.items():
        if k.endswith("conv_mu.lin.weight") or k.endswith("encoder.conv_mu.lin.weight"):
            # Shape: [out, 2*out]
            out_ch = int(v.shape[0])
            break

    return in_ch, out_ch


def build_and_load_model(in_channels, out_channels):
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels)).to(device)

    state = _safe_torch_load(model_path)
    try:
        model.load_state_dict(state)
    except Exception as e:
        print("[Error] load_state_dict failed:", type(e).__name__, e)
        print(f"Dimension Mismatch Alert: inferred in={in_channels}, out={out_channels}")
        raise

    model.eval()
    print(f"VGAE Model Loaded. (in_channels={in_channels}, out_channels={out_channels})")
    return model


@torch.no_grad()
def extract_latent_z(model, split_data, title="graph"):
    model.eval()
    z = model.encode(split_data.x, split_data.edge_index)
    print(f"\n[Latent Space Extraction] {title} | z shape = {tuple(z.shape)}")
    
    z_cpu = z.detach().cpu()
    print("Latent Vector Preview (First 5 nodes):")
    print(z_cpu[:5].numpy())
    return z


@torch.no_grad()
def eval_auc_ap(model, split_data):
    model.eval()
    z = model.encode(split_data.x, split_data.edge_index)
    # Metric Evaluation: AUC & AP (using pos/neg edge labels from RandomLinkSplit)
    auc, ap = model.test(z, split_data.pos_edge_label_index, split_data.neg_edge_label_index)
    return float(auc), float(ap)


if __name__ == "__main__":
    # 0. Check Model Existence
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found: {model_path}")
        print("   -> Please place 'best_vgae_model.pth' in the current directory.")
        raise SystemExit(1)

    print(f"Model File Verified: {model_path}")

    # 1. Data Loading & Preprocessing
    # Pipeline: CastXToFloat (Int64->Float32) -> NormalizeFeatures
    dataset = MoleculeNet(root=str(data_root), name="Lipo", transform=CastXToFloat())
    data = dataset[0]
    # Apply normalization after indexing
    data = NormalizeFeatures()(data)

    # 2. Link Split (Train/Val/Test Split for Link Prediction Task)
    splitter = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = splitter(data)

    # Transfer data to GPU
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # 3. Model Dimension Inference & Loading
    state = _safe_torch_load(model_path)
    in_ch, out_ch = infer_dims_from_state_dict(state)
    if in_ch is None or out_ch is None:
        print("[Error] Failed to infer dimensions from state_dict keys.")
        print("Key Preview:", list(state.keys())[:15])
        raise SystemExit(1)

    # Verify Feature Dimension Consistency
    x_dim = int(test_data.x.size(1))
    if x_dim != in_ch:
        print(f"[Warning] Dimension Mismatch: Dataset x_dim={x_dim} != Model in_channels={in_ch}")
        print("   -> Preprocessing mismatch detected between training and inference phases.")

    model = build_and_load_model(in_ch, out_ch)

    # 4. Inference: Latent Vector Extraction & Performance Evaluation
    extract_latent_z(model, test_data, title="Test Set")

    try:
        auc, ap = eval_auc_ap(model, test_data)
        print(f"\n[Performance Metrics] Link Prediction: AUC={auc:.4f}, AP={ap:.4f}")
    except Exception as e:
        print("\n[Warning] Metric evaluation failed:", type(e).__name__, e)
        print("   (Possible missing edge labels in split_data)")

    print("\nProcess Completed.")