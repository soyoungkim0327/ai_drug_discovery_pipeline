"""1_2_train_gat_es_p_sc.py

GAT regression on MoleculeNet: Lipo (LogP) with target scaling (StandardScaler).

Why this script exists:
- Scaling the regression target can stabilize optimization.
- The model learns to predict standardized LogP (z-score).
- At inference, predictions should be inverse-transformed back to LogP.

Artifacts written next to this script:
- best_gat_model_scaled.pth
- best_gat_model_scaled.meta.json
- logp_scaler.pkl (sklearn StandardScaler)
- logp_scaler.json (portable mean/std)

Run:
    python 01_property_prediction/1_2_train_gat_es_p_sc.py
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GATRegressor(torch.nn.Module):
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


def fit_scaler_on_train_targets(dataset, train_indices):
    """Fit StandardScaler on training targets only (avoid leakage)."""
    try:
        from sklearn.preprocessing import StandardScaler
        import joblib
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for this script (pip install scikit-learn)."
        ) from e

    y = np.array([float(dataset[i].y) for i in train_indices], dtype=np.float32).reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(y)
    return scaler


def save_scaler(scaler, out_pkl: Path, out_json: Path) -> None:
    import joblib

    joblib.dump(scaler, out_pkl)

    # portable copy (mean/std) for environments where joblib pickle is inconvenient
    meta = {
        "type": "StandardScaler",
        "mean": float(scaler.mean_[0]),
        "scale": float(scaler.scale_[0]),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def scale_y(y: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
    return (y - mean) / scale


def inverse_scale_y(y_scaled: torch.Tensor, mean: float, scale: float) -> torch.Tensor:
    return y_scaled * scale + mean


def train_one_epoch(model, loader, optimizer, device, mean: float, scale: float) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred_scaled = model(data.x.float(), data.edge_index, data.batch).view(-1)
        y_raw = data.y.view(-1)
        y_scaled = scale_y(y_raw, mean, scale)

        loss = F.mse_loss(pred_scaled, y_scaled)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_mae_in_raw_space(model, loader, device, mean: float, scale: float) -> float:
    """Compute MAE in original LogP space (inverse-transform predictions)."""
    model.eval()
    abs_err_sum = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        pred_scaled = model(data.x.float(), data.edge_index, data.batch).view(-1)
        pred_raw = inverse_scale_y(pred_scaled, mean, scale)

        y_raw = data.y.view(-1)
        abs_err_sum += float((pred_raw - y_raw).abs().sum().item())
        n += int(y_raw.numel())

    return abs_err_sum / max(1, n)


def main() -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_root = project_root / "data" / "Lipo"

    ckpt_path = script_dir / "best_gat_model_scaled.pth"
    meta_path = ckpt_path.with_suffix(".meta.json")
    scaler_pkl_path = script_dir / "logp_scaler.pkl"
    scaler_json_path = script_dir / "logp_scaler.json"

    dataset = MoleculeNet(root=str(data_root), name="Lipo")
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=gen
    )

    # Fit scaler on TRAIN only (avoid test/val leakage)
    train_indices = train_ds.indices if hasattr(train_ds, "indices") else list(range(n_train))
    scaler = fit_scaler_on_train_targets(dataset, train_indices)
    save_scaler(scaler, scaler_pkl_path, scaler_json_path)

    mean = float(scaler.mean_[0])
    scale = float(scaler.scale_[0]) if float(scaler.scale_[0]) != 0.0 else 1.0

    print(f"Dataset size: {n_total} | split train/val/test = {n_train}/{n_val}/{n_test}")
    print(f"Target scaling: mean={mean:.4f}, scale(std)={scale:.4f}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    in_dim = dataset.num_node_features
    print(f"Input node feature dim (in_dim) = {in_dim}")

    model = GATRegressor(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    patience = 20
    patience_counter = 0
    best_val_mae = float("inf")
    max_epochs = 1000

    print("Start GAT(training on scaled targets) (early stopping on VAL)...")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, mean, scale)
        val_mae = evaluate_mae_in_raw_space(model, val_loader, device, mean, scale)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_MAE(raw)={val_mae:.4f} "
                f"[best -> saved {ckpt_path.name}]"
            )
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_MAE(raw)={val_mae:.4f} "
                    f"(waiting {patience_counter}/{patience})"
                )
            if patience_counter >= patience:
                print("\nEarly stopping triggered.")
                print(f"Best VAL MAE(raw): {best_val_mae:.4f}")
                break

    # Final test evaluation (load best)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

    test_mae = evaluate_mae_in_raw_space(model, test_loader, device, mean, scale)

    meta = {
        "model_type": "GAT",
        "dataset": "MoleculeNet/Lipo",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "in_dim": in_dim,
        "target_scaling": {
            "enabled": True,
            "scaler_pkl": str(scaler_pkl_path.name),
            "scaler_json": str(scaler_json_path.name),
            "mean": mean,
            "scale": scale,
        },
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "best_val_mae": best_val_mae,
        "final_test_mae": test_mae,
        "checkpoint": str(ckpt_path.name),
    }
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Metadata written: {meta_path}")
    except Exception as e:
        print(f"[Warning] Failed to write metadata: {type(e).__name__}: {e}")

    print("\n====================")
    print(f"Final TEST MAE(raw): {test_mae:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Scaler: {scaler_pkl_path} (+ {scaler_json_path})")
    print("Done.")


if __name__ == "__main__":
    main()
