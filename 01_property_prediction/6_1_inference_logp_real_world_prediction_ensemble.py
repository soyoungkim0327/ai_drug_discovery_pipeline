"""
Hybrid Inference (Similarity-Gated, Production-ish)
===================================================
- Uses train-distribution proxy (Max Tanimoto similarity) as a deployable gate.
- Caches Lipo fingerprints for fast repeated runs.
- Infers model input dim / heads / hidden / concat from checkpoint state_dict.
- Enforces feature extractor consistency (from_smiles first; dim mismatch -> explicit error).

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-03
"""

import os
import re
import json
import joblib
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

# -----------------------------
# [Configuration]
# -----------------------------
RDLogger.DisableLog("rdApp.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Computational Device: {device}")

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_A_FILE = "best_gat_model.pth"
MODEL_B_FILE = "best_gat_model_dropout.pth"
SCALER_FILE  = "logp_scaler.pkl"

SIMILARITY_THRESHOLD = 0.50

# Fingerprint DB cache
FP_CACHE_FILE = "_cache_lipo_fps_r2_1024.pkl"
FP_RADIUS = 2
FP_NBITS  = 1024


# -----------------------------
# 1) Robust CSV Finder + Fingerprint DB (Cached)
# -----------------------------
def find_lipophilicity_csv() -> Path | None:
    """
    Tries common PyG MoleculeNet folder layouts first,
    then falls back to searching upward (bounded) for 'Lipophilicity.csv'.
    """
    candidates = [
        SCRIPT_DIR / ".." / "data" / "Lipo" / "lipo" / "raw" / "Lipophilicity.csv",
        SCRIPT_DIR / ".." / "data" / "Lipo" / "raw" / "Lipophilicity.csv",
        SCRIPT_DIR / "data" / "Lipo" / "lipo" / "raw" / "Lipophilicity.csv",
        SCRIPT_DIR / "data" / "Lipo" / "raw" / "Lipophilicity.csv",
    ]
    for p in candidates:
        p = p.resolve()
        if p.exists():
            return p

    # bounded upward search (avoid scanning entire disk)
    for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents)[:6]:
        hits = list(parent.rglob("Lipophilicity.csv"))
        if hits:
            return hits[0].resolve()
    return None


def _detect_smiles_column(df: pd.DataFrame) -> str:
    for c in ["smiles", "SMILES"]:
        if c in df.columns:
            return c
    # fallback: find a column containing "smiles" case-insensitive
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    raise ValueError(f"SMILES column not found. Columns={list(df.columns)}")


def load_training_fingerprints_cached() -> list:
    """
    Returns a list of Morgan fingerprints built from Lipophilicity.csv.
    Cached to SCRIPT_DIR/FP_CACHE_FILE for repeatability & speed.
    """
    cache_path = SCRIPT_DIR / FP_CACHE_FILE
    if cache_path.exists():
        try:
            fps = joblib.load(cache_path)
            if isinstance(fps, list) and len(fps) > 0:
                print(f"[*] Fingerprint cache loaded: {cache_path.name} ({len(fps)} mols)")
                return fps
        except Exception:
            pass  # rebuild on any cache issue

    csv_path = find_lipophilicity_csv()
    if not csv_path:
        print("[!] Warning: Lipophilicity.csv not found. Similarity gating will be disabled (max_sim=0).")
        return []

    print(f"[*] Building fingerprint DB from: {csv_path}")
    df = pd.read_csv(csv_path)
    col = _detect_smiles_column(df)

    fps = []
    for s in df[col].astype(str).tolist():
        mol = Chem.MolFromSmiles(s)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
            fps.append(fp)

    print(f"[*] Memory Built: {len(fps)} molecules indexed.")
    if len(fps) > 0:
        joblib.dump(fps, cache_path)
        print(f"[*] Fingerprint cache saved: {cache_path.name}")

    return fps


def get_max_similarity(target_smiles: str, db_fingerprints: list) -> float:
    if not db_fingerprints:
        return 0.0
    mol = Chem.MolFromSmiles(target_smiles)
    if not mol:
        return 0.0
    target_fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    sims = DataStructs.BulkTanimotoSimilarity(target_fp, db_fingerprints)
    return float(max(sims)) if sims else 0.0


# -----------------------------
# 2) Robust GAT/GATv2 Loader (Infer Architecture from state_dict)
# -----------------------------
def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _detect_conv_type(state: dict) -> str:
    # Heuristic: GATv2Conv often has "lin_l" / "lin_r" params
    keys = state.keys()
    if any("lin_l.weight" in k for k in keys) or any("lin_r.weight" in k for k in keys):
        return "gatv2"
    return "gat"


def _infer_heads_hidden_concat_in_dim(state: dict) -> dict:
    """
    Tries to infer:
      - in_dim
      - heads
      - hidden_dim (per-head out_channels)
      - concat
    from conv1 parameters.
    """
    keys = list(state.keys())

    # 1) in_dim from conv1 linear weight
    w_key_candidates = [
        "conv1.lin_src.weight",
        "conv1.lin.weight",
        "conv1.lin_l.weight",
    ]
    in_dim = None
    for k in w_key_candidates:
        if k in state:
            in_dim = int(state[k].shape[1])
            break
    if in_dim is None:
        # fallback: find any conv1.*weight matrix
        for k in keys:
            if k.startswith("conv1.") and k.endswith(".weight") and state[k].ndim == 2:
                in_dim = int(state[k].shape[1])
                break
    if in_dim is None:
        raise RuntimeError("Failed to infer in_dim from checkpoint. (conv1 weight not found)")

    # 2) heads & hidden_dim from attention parameter
    att_key_candidates = ["conv1.att_src", "conv1.att_l", "conv1.att"]
    heads = None
    hidden = None
    for k in att_key_candidates:
        if k in state:
            t = state[k]
            if t.ndim == 3:
                # common: (1, heads, out_channels)
                heads = int(t.shape[1])
                hidden = int(t.shape[2])
            elif t.ndim == 2:
                heads = int(t.shape[0])
                hidden = int(t.shape[1])
            break

    # if attention not found, try bias shape heuristics
    if heads is None or hidden is None:
        # infer hidden from conv2.bias if present
        if "conv2.bias" in state:
            hidden = int(state["conv2.bias"].numel())
        else:
            hidden = 64  # safe fallback
        heads = 4      # safe fallback

    # 3) concat from conv1.bias size (if available)
    concat = False
    if "conv1.bias" in state:
        bsz = int(state["conv1.bias"].numel())
        if bsz == heads * hidden:
            concat = True
        elif bsz == hidden:
            concat = False
        else:
            # ambiguous; default False
            concat = False

    return {"in_dim": in_dim, "heads": heads, "hidden": hidden, "concat": concat}


def _infer_num_layers(state: dict) -> int:
    # conv1/conv2/conv3 ... style
    conv_ids = set()
    for k in state.keys():
        m = re.match(r"conv(\d+)\.", k)
        if m:
            conv_ids.add(int(m.group(1)))
    return max(conv_ids) if conv_ids else 3


def build_gat_regressor(conv_type: str, in_dim: int, hidden: int, heads: int, concat: bool, num_layers: int):
    try:
        if conv_type == "gatv2":
            from torch_geometric.nn import GATv2Conv as Conv
        else:
            from torch_geometric.nn import GATConv as Conv
    except Exception as e:
        raise RuntimeError(f"Failed to import conv ({conv_type}). Install a compatible torch-geometric. Error={e}")

    class GATRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Input -> hidden
            self.conv1 = Conv(in_dim, hidden, heads=heads, concat=concat)
            # hidden -> hidden
            in2 = hidden * heads if concat else hidden
            self.conv2 = Conv(in2, hidden, heads=heads, concat=concat)
            in3 = hidden * heads if concat else hidden
            self.conv3 = Conv(in3, hidden, heads=heads, concat=concat)
            out_dim = hidden * heads if concat else hidden
            self.lin = torch.nn.Linear(out_dim, 1)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            x = self.conv3(x, edge_index).relu()
            x = global_mean_pool(x, batch)
            return self.lin(x)

    return GATRegressor()


def load_model(model_file: str):
    model_path = (SCRIPT_DIR / model_file).resolve()
    if not model_path.exists():
        return None, None

    ckpt = torch.load(model_path, map_location=device)
    state = _unwrap_state_dict(ckpt)

    conv_type = _detect_conv_type(state)
    meta = _infer_heads_hidden_concat_in_dim(state)
    num_layers = _infer_num_layers(state)

    model = build_gat_regressor(
        conv_type=conv_type,
        in_dim=meta["in_dim"],
        hidden=meta["hidden"],
        heads=meta["heads"],
        concat=meta["concat"],
        num_layers=num_layers
    )

    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    arch = {"conv_type": conv_type, **meta, "num_layers": num_layers}
    return model, arch


# -----------------------------
# 3) Graph Builder (Feature Extractor Consistency)
# -----------------------------
def smiles_to_graph(smiles: str, expected_in_dim: int) -> Data | None:
    try:
        data = from_smiles(smiles)  # must match training featurizer
        data.x = data.x.to(torch.float)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        if data.x.size(1) != expected_in_dim:
            raise ValueError(
                f"Feature dim mismatch: from_smiles gives {data.x.size(1)} but model expects {expected_in_dim}. "
                "Fix: align torch_geometric version/featurizer with training."
            )
        return data
    except Exception as e:
        print(f"[!] Graph build failed for SMILES={smiles} | Error={e}")
        return None


# -----------------------------
# 4) Main
# -----------------------------
if __name__ == "__main__":
    print("\n=== Hybrid Inference: Similarity-Gated (A=Expert, B=Safety) ===")

    # 1) Fingerprint DB
    db_fps = load_training_fingerprints_cached()

    # 2) Scaler
    scaler_path = (SCRIPT_DIR / SCALER_FILE).resolve()
    if not scaler_path.exists():
        # auto-detect
        candidates = [p for p in SCRIPT_DIR.glob("*.pkl") if "scaler" in p.name.lower()]
        if candidates:
            scaler_path = candidates[0].resolve()

    if not scaler_path.exists():
        raise FileNotFoundError("No scaler found in script directory.")
    scaler = joblib.load(scaler_path)
    print(f"[*] Scaler: {scaler_path.name}")

    # 3) Models
    model_a, arch_a = load_model(MODEL_A_FILE)
    model_b, arch_b = load_model(MODEL_B_FILE)

    if model_a is None or model_b is None:
        raise FileNotFoundError("Model A or B checkpoint not found in script directory.")

    # sanity: input dim must match
    if arch_a["in_dim"] != arch_b["in_dim"]:
        raise RuntimeError(f"Model A/B in_dim mismatch: {arch_a['in_dim']} vs {arch_b['in_dim']}")

    in_dim = arch_a["in_dim"]
    print(f"[*] Model A arch: {arch_a}")
    print(f"[*] Model B arch: {arch_b}")

    molecules = {
        "Aspirin":  "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Tylenol":  "CC(=O)NC1=CC=C(C=C1)O",
        "Benzene":  "C1=CC=CC=C1",
        "Sugar":    "C(C1C(C(C(C(O1)O)O)O)O)O",
    }

    print("\n" + "=" * 140)
    print(f"{'Molecule':<12} | {'MaxSim':<8} | {'PredA':<10} | {'PredB':<10} | {'Final':<10} | {'Decision':<35}")
    print("-" * 140)

    for name, smi in molecules.items():
        max_sim = get_max_similarity(smi, db_fps)

        # Build graph
        data = smiles_to_graph(smi, expected_in_dim=in_dim)
        if data is None:
            continue

        data = data.to(device)
        with torch.no_grad():
            out_a = model_a(data.x.float(), data.edge_index, data.batch).item()
            out_b = model_b(data.x.float(), data.edge_index, data.batch).item()

        pred_a = float(scaler.inverse_transform([[out_a]])[0][0])
        pred_b = float(scaler.inverse_transform([[out_b]])[0][0])

        if max_sim >= SIMILARITY_THRESHOLD:
            final = pred_a
            decision = f"A (Known, sim={max_sim:.2f})"
        else:
            final = pred_b
            decision = f"B (Unknown, sim={max_sim:.2f})"

        print(f"{name:<12} | {max_sim:<8.4f} | {pred_a:<10.4f} | {pred_b:<10.4f} | {final:<10.4f} | {decision:<35}")

    print("=" * 140)
    print(f"[*] Gate: if MaxSim >= {SIMILARITY_THRESHOLD}, use A else B.")
