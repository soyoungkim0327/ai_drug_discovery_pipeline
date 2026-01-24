"""2_train_schnet_3d_es_p.py

SchNet 3D regression on a 3D-conformer-augmented version of MoleculeNet(Lipo).

Changes (professional hygiene):
- Robust paths (project_root/data)
- Proper train/val/test split (early stopping monitors VAL)
- main() entry-point
- Saves checkpoint next to this script + small metadata JSON

Run:
    python 02_3d_structure/2_train_schnet_3d_es_p.py
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch_geometric.datasets import MoleculeNet

from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Lipophilicity3D(InMemoryDataset):
    """Build a small 3D dataset by embedding a subset of Lipo molecules."""

    def __init__(self, root: str, raw_dataset, max_mols: int = 500, transform=None):
        self.raw_dataset = raw_dataset
        self.max_mols = max_mols
        super().__init__(root, transform)

        # Compatibility: torch.load weights_only kwarg exists only in newer PyTorch
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data_3d.pt"]

    def process(self):
        data_list = []
        n_fail = 0
        print("Converting 2D SMILES -> 3D conformers...")

        for i in tqdm(range(min(self.max_mols, len(self.raw_dataset)))):
            item = self.raw_dataset[i]
            mol = Chem.MolFromSmiles(getattr(item, "smiles", None))
            if mol is None:
                n_fail += 1
                continue

            mol = Chem.AddHs(mol)
            try:
                # 3D embedding + forcefield optimization
                params = AllChem.ETKDGv3()
                if AllChem.EmbedMolecule(mol, params) == -1:
                    n_fail += 1
                    continue
                AllChem.MMFFOptimizeMolecule(mol)
                conf = mol.GetConformer()
            except Exception:
                n_fail += 1
                continue

            pos = []
            z = []
            for atom in mol.GetAtoms():
                z.append(atom.GetAtomicNum())
                pos.append(list(conf.GetAtomPosition(atom.GetIdx())))

            data = Data(
                z=torch.tensor(z, dtype=torch.long),
                pos=torch.tensor(pos, dtype=torch.float),
                y=item.y,
            )
            data_list.append(data)

        print(f"3D build finished: kept={len(data_list)}, failed={n_fail}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


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


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data.z, data.pos, data.batch).view(-1)
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
        pred = model(data.z, data.pos, data.batch).view(-1)
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
    lipo_root = project_root / "data" / "Lipo"
    lipo3d_root = project_root / "data" / "Lipo_3D"

    ckpt_path = script_dir / "best_schnet_model.pth"
    meta_path = ckpt_path.with_suffix(".meta.json")

    raw_data = MoleculeNet(root=str(lipo_root), name="Lipo")
    dataset_3d = Lipophilicity3D(root=str(lipo3d_root), raw_dataset=raw_data, max_mols=500)

    n_total = len(dataset_3d)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset_3d, [n_train, n_val, n_test], generator=gen
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"3D dataset size: {n_total} | split train/val/test = {n_train}/{n_val}/{n_test}")

    model = SchNetModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    patience = 10
    patience_counter = 0
    best_val_mae = float("inf")
    max_epochs = 200

    print("Start 3D SchNet training (early stopping on VAL)...")

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
            if epoch % 5 == 0:
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

    meta = {
        "model_type": "SchNet",
        "dataset": "MoleculeNet/Lipo (3D embedded subset)",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
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
