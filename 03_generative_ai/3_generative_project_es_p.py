"""3_generative_project_es_p.py

VGAE (Variational Graph AutoEncoder) for link prediction on a molecule graph.

NOTE:
- This script uses a single molecular graph (dataset[0]) as a compact demo.
- Early stopping monitors VAL (not TEST) to avoid evaluation leakage.

Artifacts written next to this script:
- best_vgae_model.pth
- best_vgae_model.meta.json

Run:
    python 03_generative_ai/3_generative_project_es_p.py
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, VGAE


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def train_one_epoch(model, optimizer, train_data) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + (1.0 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def eval_auc_ap(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


def main() -> None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    seed = 42
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_root = project_root / "data" / "Lipo"

    ckpt_path = script_dir / "best_vgae_model.pth"
    meta_path = ckpt_path.with_suffix(".meta.json")

    # 1) Data
    dataset = MoleculeNet(root=str(data_root), name="Lipo")

    # Demo: take a single molecular graph
    data = dataset[0]
    data.x = data.x.float()

    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = transform(data)

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # 2) Model
    out_channels = 16
    model = VGAE(VariationalGCNEncoder(dataset.num_features, out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3) Early stopping on VAL AUC
    patience = 20
    patience_counter = 0
    best_val_auc = -1.0
    max_epochs = 1000

    print("Start VGAE training (early stopping on VAL AUC)...")

    for epoch in range(1, max_epochs + 1):
        loss = train_one_epoch(model, optimizer, train_data)
        val_auc, val_ap = eval_auc_ap(model, val_data)

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"Epoch {epoch:03d} | loss={loss:.4f} | val_auc={val_auc:.4f} | val_ap={val_ap:.4f} "
                f"[best -> saved {ckpt_path.name}]"
            )
        else:
            patience_counter += 1
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:03d} | loss={loss:.4f} | val_auc={val_auc:.4f} | val_ap={val_ap:.4f} "
                    f"(waiting {patience_counter}/{patience})"
                )
            if patience_counter >= patience:
                print("\nEarly stopping triggered.")
                print(f"Best VAL AUC: {best_val_auc:.4f}")
                break

    # 4) Final TEST evaluation
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

    test_auc, test_ap = eval_auc_ap(model, test_data)

    meta = {
        "model_type": "VGAE",
        "dataset": "MoleculeNet/Lipo (single-graph demo: dataset[0])",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "out_channels": out_channels,
        "split": {"val": 0.05, "test": 0.10, "train": 0.85},
        "best_val_auc": best_val_auc,
        "final_test_auc": float(test_auc),
        "final_test_ap": float(test_ap),
        "checkpoint": str(ckpt_path.name),
    }
    try:
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Metadata written: {meta_path}")
    except Exception as e:
        print(f"[Warning] Failed to write metadata: {type(e).__name__}: {e}")

    print("\n====================")
    print(f"Final TEST: AUC={test_auc:.4f} | AP={test_ap:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print("Done.")


if __name__ == "__main__":
    main()
