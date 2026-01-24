"""1_1_train_gcn_es_p.py

GCN baseline regression (LogP) on MoleculeNet: Lipo.

Changes (professional hygiene):
- Proper train/val/test split (early stopping monitors VAL, not TEST).
- Robust paths: dataset + checkpoints resolve from project root.
- main() entry-point.

Run:
    python 01_property_prediction/1_1_train_gcn_es_p.py
"""

import os
import json
from datetime import datetime
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GCNRegressor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data.x.float(), data.edge_index, data.batch).view(-1)
        y = data.y.view(-1)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_mae(model, loader, device) -> float:
    model.eval()
    abs_err_sum = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x.float(), data.edge_index, data.batch).view(-1)
        y = data.y.view(-1)
        abs_err_sum += float((pred - y).abs().sum().item())
        n += int(y.numel())
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
    ckpt_path = script_dir / "best_gcn_model.pth"
    meta_path = ckpt_path.with_suffix(".meta.json")

    dataset = MoleculeNet(root=str(data_root), name="Lipo")
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=gen
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    in_dim = dataset.num_node_features
    print(f"Dataset size: {n_total} | split train/val/test = {n_train}/{n_val}/{n_test}")
    print(f"Input node feature dim (in_dim) = {in_dim}")

    model = GCNRegressor(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    patience = 20
    patience_counter = 0
    best_val_mae = float("inf")
    max_epochs = 1000

    print("Start GCN training (early stopping on VAL)...")

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_mae = evaluate_mae(model, val_loader, device)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_MAE={val_mae:.4f} "
                f"[best -> saved {ckpt_path.name}]"
            )
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_MAE={val_mae:.4f} "
                    f"(waiting {patience_counter}/{patience})"
                )
            if patience_counter >= patience:
                print("\nEarly stopping triggered.")
                print(f"Best VAL MAE: {best_val_mae:.4f}")
                break

    # Final test evaluation
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

    test_mae = evaluate_mae(model, test_loader, device)

    # Write lightweight metadata for inference/debugging
    meta = {
        "model_type": "GCN",
        "dataset": "MoleculeNet/Lipo",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "in_dim": in_dim,
        "target_scaling": {"enabled": False},
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
    print(f"Final TEST MAE: {test_mae:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print("Done.")


if __name__ == "__main__":
    main()
