"""
Calibration Script: Find Best Gate Threshold (Robust & Auto-Detect)
===================================================================
"Finds the optimal similarity threshold that minimizes RMSE on a validation set."
Now includes robust model architecture inference (same as inference script).

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-03
"""

import os
import re
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.utils import from_smiles
from torch_geometric.nn import global_mean_pool
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from rdkit import RDLogger

# [Config]
RDLogger.DisableLog("rdApp.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_A_FILE = "best_gat_model.pth"
MODEL_B_FILE = "best_gat_model_dropout.pth"
SCALER_FILE  = "logp_scaler.pkl"
FP_RADIUS = 2
FP_NBITS  = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1. Helper Functions (File & Fingerprint)
# -----------------------------
def find_file(filename):
    p = SCRIPT_DIR / filename
    if p.exists(): return p
    for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents)[:4]:
        hits = list(parent.rglob(filename))
        if hits: return hits[0]
    return None

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol: return AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    return None

def get_max_sim(target_fp, db_fps):
    if not db_fps or target_fp is None: return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(target_fp, db_fps)
    return max(sims)

# -----------------------------
# 2. Robust Model Loader (핵심 수정 부분)
# -----------------------------
def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def _detect_conv_type(state: dict) -> str:
    keys = state.keys()
    if any("lin_l.weight" in k for k in keys) or any("lin_r.weight" in k for k in keys):
        return "gatv2"
    return "gat"

def _infer_heads_hidden_concat_in_dim(state: dict) -> dict:
    keys = list(state.keys())
    
    # In_dim inference
    in_dim = None
    w_key_candidates = ["conv1.lin_src.weight", "conv1.lin.weight", "conv1.lin_l.weight"]
    for k in w_key_candidates:
        if k in state:
            in_dim = int(state[k].shape[1])
            break
    if in_dim is None:
        for k in keys:
            if k.startswith("conv1.") and k.endswith(".weight") and state[k].ndim == 2:
                in_dim = int(state[k].shape[1])
                break
    if in_dim is None: raise RuntimeError("Cannot infer input dim from checkpoint")

    # Heads & Hidden inference
    att_key_candidates = ["conv1.att_src", "conv1.att_l", "conv1.att"]
    heads = None
    hidden = None
    for k in att_key_candidates:
        if k in state:
            t = state[k]
            if t.ndim == 3:
                heads = int(t.shape[1])
                hidden = int(t.shape[2])
            elif t.ndim == 2:
                heads = int(t.shape[0])
                hidden = int(t.shape[1])
            break
            
    if heads is None or hidden is None:
        if "conv2.bias" in state: hidden = int(state["conv2.bias"].numel())
        else: hidden = 64
        heads = 4

    # Concat inference
    concat = False
    if "conv1.bias" in state:
        bsz = int(state["conv1.bias"].numel())
        if bsz == heads * hidden: concat = True
    
    return {"in_dim": in_dim, "heads": heads, "hidden": hidden, "concat": concat}

def _infer_num_layers(state: dict) -> int:
    conv_ids = set()
    for k in state.keys():
        m = re.match(r"conv(\d+)\.", k)
        if m: conv_ids.add(int(m.group(1)))
    return max(conv_ids) if conv_ids else 3

def build_gat_regressor(conv_type, in_dim, hidden, heads, concat, num_layers):
    try:
        if conv_type == "gatv2": from torch_geometric.nn import GATv2Conv as Conv
        else: from torch_geometric.nn import GATConv as Conv
    except: from torch_geometric.nn import GATConv as Conv

    class GATRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv(in_dim, hidden, heads=heads, concat=concat)
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

def load_robust_model(filename):
    path = find_file(filename)
    if not path:
        print(f"[!] Model not found: {filename}")
        return None, None
        
    ckpt = torch.load(path, map_location=device)
    state = _unwrap_state_dict(ckpt)
    
    conv_type = _detect_conv_type(state)
    meta = _infer_heads_hidden_concat_in_dim(state)
    num_layers = _infer_num_layers(state)
    
    model = build_gat_regressor(conv_type, meta["in_dim"], meta["hidden"], meta["heads"], meta["concat"], num_layers)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    
    print(f"[*] Loaded {filename}: {meta}")
    return model, meta["in_dim"]

# -----------------------------
# 3. Main Calibration
# -----------------------------
if __name__ == "__main__":
    print("\n[*] Starting Threshold Calibration (Robust Mode)...")
    
    # 1. Load Data
    csv_path = find_file("Lipophilicity.csv")
    if not csv_path: raise FileNotFoundError("Lipophilicity.csv not found")
    print(f"[*] Data: {csv_path.name}")
    
    df = pd.read_csv(csv_path)
    smi_col = [c for c in df.columns if 'smiles' in c.lower()][0]
    y_col = [c for c in df.columns if 'exp' in c.lower() or 'log' in c.lower()][0]
    
    # 2. Load Scaler
    scaler_path = find_file(SCALER_FILE)
    if not scaler_path: raise FileNotFoundError("Scaler not found")
    scaler = joblib.load(scaler_path)
    
    # 3. Load Models
    model_a, dim_a = load_robust_model(MODEL_A_FILE)
    model_b, dim_b = load_robust_model(MODEL_B_FILE)
    if not model_a or not model_b: exit()
    if dim_a != dim_b: raise RuntimeError(f"Input dim mismatch: {dim_a} vs {dim_b}")
    in_dim = dim_a

    # 4. Split & DB
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"[*] Split: Train {len(train_df)} | Val {len(val_df)}")
    print("[*] Building Train Fingerprints...")
    train_fps = [get_fingerprint(s) for s in train_df[smi_col] if get_fingerprint(s)]

    # 5. Pre-calculate
    print("[*] Pre-calculating predictions...")
    val_results = []
    
    with torch.no_grad():
        for _, row in val_df.iterrows():
            smi = row[smi_col]
            true_val = row[y_col]
            
            # Graph (Consistent Featurizer)
            try:
                data = from_smiles(smi)
                data.x = data.x.float()
                if data.x.size(1) != in_dim: continue # Mismatch skip
                data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
                data = data.to(device)
            except: continue
            
            p_a = model_a(data.x, data.edge_index, data.batch).item()
            p_b = model_b(data.x, data.edge_index, data.batch).item()
            
            val_results.append({
                'true': true_val,
                'pred_a': scaler.inverse_transform([[p_a]])[0][0],
                'pred_b': scaler.inverse_transform([[p_b]])[0][0],
                'sim': get_max_sim(get_fingerprint(smi), train_fps)
            })

    # 6. Grid Search
    print("\n[*] Running Grid Search...")
    results_df = pd.DataFrame(val_results)
    best_rmse = float('inf')
    best_thr = 0.0
    
    print(f"{'Threshold':<10} | {'RMSE':<12} | {'A Usage (%)':<12}")
    print("-" * 40)
    
    for thr in np.arange(0.0, 1.01, 0.05):
        hyb = []
        cnt_a = 0
        for _, r in results_df.iterrows():
            if r['sim'] >= thr:
                hyb.append(r['pred_a'])
                cnt_a += 1
            else:
                hyb.append(r['pred_b'])
        
        rmse = np.sqrt(mean_squared_error(results_df['true'], hyb))
        print(f"{thr:<10.2f} | {rmse:<12.4f} | {(cnt_a/len(results_df)*100):<12.1f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_thr = thr
            
    print("-" * 40)
    print(f"[🏆 FINAL] Best Threshold: {best_thr:.2f} (RMSE: {best_rmse:.4f})")
    print(f"Action: Update SIMILARITY_THRESHOLD in your inference script.")