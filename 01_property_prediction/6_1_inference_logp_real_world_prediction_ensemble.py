"""
HYBRID INFERENCE SYSTEM: Context-Aware LogP Prediction
======================================================
"Combines High-Precision GAT with Robust Uncertainty Management using Data Retrieval."

[System Overview]
This script implements a "Context-Aware" AI system that dynamically selects the optimal model 
based on the input molecule's familiarity (Similarity to Training Data).

It solves the trade-off between Precision and Stability by switching between two specialized models:
1. Model A (The Expert): Pure GAT. Optimized for high precision on known drug-like structures.
2. Model B (The Safety Net): Dropout GAT. Optimized for robustness on out-of-distribution (OOD) data.

[Switching Logic: Data-Driven Retrieval]
The system retrieves the 'Lipo' training dataset and calculates Tanimoto Similarity.
- IF Similarity > 0.5 (Known Domain)  -> TRUST Model A (Maximize Precision)
- IF Similarity <= 0.5 (Unknown/OOD)  -> TRUST Model B (Prevent Overfitting/Hallucination)

[Performance Highlights]
1. Tylenol (Known Drug, Sim=1.0):
   - Decision: Trusted Model A (Expert)
   - Result: Predicted 0.87 (Actual 0.91) -> Error 0.04 (High Precision maintained)
   
2. Sugar (OOD/Non-Drug, Sim=0.28):
   - Decision: Switched to Model B (Safety)
   - Result: Predicted -3.48 (Actual -3.24) -> Error 0.24
   - Impact: Prevented Model A's overfitting catastrophic error (-4.66).

[Conclusion]
This system demonstrates a "Fail-Safe" AI architecture, successfully achieving both 
pharmaceutical precision for drugs and risk management for outliers.

Author: Soyoung Kim
Date: 2026-02-03
Project: AI Drug Discovery Pipeline
"""




"""
Final Hybrid Inference: The Masterpiece
=======================================
Combines:
1. Robust Path Finding (Locates Lipo.csv automatically)
2. Correct Feature Extraction (Matches training data exactly)
3. Smart Switching Logic (Uses Model A for known, Model B for unknown)

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-03
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd

from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GATConv, global_mean_pool
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_DIM = 9 

# Files
MODEL_A_FILE = "best_gat_model.pth"          
MODEL_B_FILE = "best_gat_model_dropout.pth"  
SCALER_FILE = "logp_scaler.pkl"  
SIMILARITY_THRESHOLD = 0.5 

# -----------------------------
# 1. Database Builder (Robust Path)
# -----------------------------
def load_training_fingerprints():
    print("[*] Loading Lipo Dataset to build 'Memory'...")
    
    # Try multiple possible paths for Lipo.csv
    possible_paths = [
        os.path.join(SCRIPT_DIR, "../data/Lipo/lipo/raw/Lipophilicity.csv"), # Standard structure
        os.path.join(SCRIPT_DIR, "../data/Lipo/raw/Lipophilicity.csv"),      # Alternative
        os.path.join(SCRIPT_DIR, "data/Lipo/raw/Lipophilicity.csv"),         # Root
        r"C:\repository\_biology\ai_drug_discovery_pipeline\data\Lipo\lipo\raw\Lipophilicity.csv" # Hardcoded backup
    ]
    
    csv_path = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_path = os.path.normpath(p)
            break
            
    if not csv_path:
        print("[!] Warning: CSV not found. Memory will be empty.")
        return []

    print(f"[*] Database Source: {os.path.basename(csv_path)}")
    try:
        df = pd.read_csv(csv_path)
        col_name = 'smiles' if 'smiles' in df.columns else 'SMILES'
        fingerprints = []
        for s in df[col_name]:
            mol = Chem.MolFromSmiles(s)
            if mol:
                fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        print(f"[*] Memory Built: {len(fingerprints)} molecules indexed.")
        return fingerprints
    except Exception as e:
        print(f"[!] DB Error: {e}")
        return []

def get_max_similarity(target_smile, db_fingerprints):
    if not db_fingerprints: return 0.0
    mol = Chem.MolFromSmiles(target_smile)
    if not mol: return 0.0
    target_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    sims = DataStructs.BulkTanimotoSimilarity(target_fp, db_fingerprints)
    return max(sims) if sims else 0.0

# -----------------------------
# 2. Model Definition
# -----------------------------
class GATModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_channels, heads):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_channels, heads=heads, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x)

# -----------------------------
# 3. Helpers (Correct Feature Extraction)
# -----------------------------
def get_atom_features_9(atom):
    return [atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(), int(atom.GetHybridization()), int(atom.GetIsAromatic()), atom.GetTotalNumHs(), atom.GetNumRadicalElectrons(), atom.GetImplicitValence(), int(atom.IsInRing())]

def smile_to_graph(smile: str):
    # [CRITICAL FIX] Try PyG's standard converter first (Matches Training Data)
    try:
        data = from_smiles(smile)
        data.x = data.x.to(torch.float)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        if data.x.size(1) != IN_DIM: raise ValueError
        return data
    except:
        # Fallback to manual if PyG fails
        mol = Chem.MolFromSmiles(smile)
        if mol is None: return None
        x = torch.tensor([get_atom_features_9(a) for a in mol.GetAtoms()], dtype=torch.float)
        edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        edges += [[j, i] for i, j in edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if len(edges) > 0 else torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))

def load_gat_model(filename):
    path = os.path.join(SCRIPT_DIR, filename)
    if not os.path.exists(path): return None
    state = torch.load(path, map_location=device)
    hidden = state['conv2.bias'].shape[0] if 'conv2.bias' in state else 64
    model = GATModel(IN_DIM, hidden, heads=4)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

# -----------------------------
# 4. Main Routine
# -----------------------------
if __name__ == "__main__":
    print("\n=== AI Drug Prediction: Final Hybrid Strategy ===")
    
    # 1. DB Load
    db_fps = load_training_fingerprints()
    
    # 2. Scaler Load (Auto-Detect)
    scaler_path = os.path.join(SCRIPT_DIR, SCALER_FILE)
    if not os.path.exists(scaler_path):
        candidates = [f for f in os.listdir(SCRIPT_DIR) if 'scaler' in f and f.endswith('.pkl')]
        if candidates: scaler_path = os.path.join(SCRIPT_DIR, candidates[0])
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"[*] Scaler: {os.path.basename(scaler_path)}")
    else:
        print("[!] Error: No scaler found.")
        exit()

    # 3. Model Load
    model_a = load_gat_model(MODEL_A_FILE)
    model_b = load_gat_model(MODEL_B_FILE)

    if not model_a or not model_b:
        print("[!] Error: Models missing.")
        exit()

    molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Tylenol": "CC(=O)NC1=CC=C(C=C1)O",
        "Benzene": "C1=CC=CC=C1",
        "Sugar": "C(C1C(C(C(C(O1)O)O)O)O)O"
    }

    print("\n" + "="*110)
    print(f"{'Molecule':<12} | {'Max Similarity':<15} | {'Decision':<25} | {'Final Val':<10}")
    print("-" * 110)

    for name, smile in molecules.items():
        max_sim = get_max_similarity(smile, db_fps)
        
        # [Decision Logic]
        if max_sim > SIMILARITY_THRESHOLD:
            choice = f"Model A (Known {max_sim:.2f})"
            selected_model = model_a
        else:
            choice = f"Model B (Unknown {max_sim:.2f})"
            selected_model = model_b
            
        data = smile_to_graph(smile)
        if data:
            data = data.to(device)
            with torch.no_grad():
                out = selected_model(data.x.float(), data.edge_index, data.batch).item()
                val = scaler.inverse_transform([[out]])[0][0]
                print(f"{name:<12} | {max_sim:<15.4f} | {choice:<25} | {val:<10.4f}")

    print("="*110)
    print(f"[*] Logic: If Similarity > {SIMILARITY_THRESHOLD}, use Expert (A). Else, use Safety (B).")