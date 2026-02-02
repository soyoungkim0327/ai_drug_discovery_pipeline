"""
SchNet 3D Inference Script (QM9 / Energy U0)
============================================
This script performs inference using a pre-trained SchNet model trained on the QM9 dataset.
It predicts the Internal Energy at 0K (U0) for any given molecule (SMILES).

Process:
1. Takes a 2D SMILES string as input.
2. Generates a 3D conformer using RDKit (Embedding + MMFF Optimization).
3. Extracts atomic numbers (z) and 3D positions (pos).
4. Predicts the scaled energy using the SchNet model.
5. Inverse-transforms the prediction to retrieve the real physical value (U0).

Dependencies:
- torch_geometric (SchNet)
- rdkit (Conformer Generation)
- sklearn (Scaler Inverse Transform)

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-03
"""

import os
import torch
import joblib
import numpy as np
from torch_geometric.nn import SchNet
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# -----------------------------------------------------------------------------
# 1. System & Environment Configuration
# -----------------------------------------------------------------------------
# Disable RDKit internal warnings for cleaner output
RDLogger.DisableLog("rdApp.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Computational Device Allocation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Computational Device: {device}")

# Path Configuration (Relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "best_schnet_qm9_model.pth"
SCALER_FILE = "qm9_u0_scaler.pkl"

MODEL_PATH = os.path.join(SCRIPT_DIR, MODEL_FILE)
SCALER_PATH = os.path.join(SCRIPT_DIR, SCALER_FILE)


# -----------------------------------------------------------------------------
# 2. Model Architecture Definition
# -----------------------------------------------------------------------------
class SchNetRegressor(torch.nn.Module):
    """
    SchNet Regressor Wrapper.
    
    Must strictly match the architecture defined in the training script
    'SchNet QM9 Benchmark Training'.
    
    Configuration:
    - Hidden Channels: 64
    - Interactions: 3
    - Cutoff: 10.0
    """
    def __init__(self):
        super().__init__()
        self.model = SchNet(
            hidden_channels=64,
            num_filters=64,
            num_interactions=3,
            num_gaussians=50,
            cutoff=10.0,
        )

    def forward(self, z, pos, batch):
        # SchNet returns [batch_size, 1] output automatically
        return self.model(z, pos, batch)


# -----------------------------------------------------------------------------
# 3. Initialization & Loading Utils
# -----------------------------------------------------------------------------
def load_system():
    """
    Loads the trained model checkpoint and the data scaler.
    """
    # 1. Validation
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[Error] Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"[Error] Scaler file not found: {SCALER_PATH}")

    # 2. Load Model
    print(f"[*] Loading Model from: {os.path.basename(MODEL_PATH)}")
    model = SchNetRegressor().to(device)
    
    # Robust state_dict loading (handling potential weights_only pickle issues)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load Scaler
    print(f"[*] Loading Scaler from: {os.path.basename(SCALER_PATH)}")
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# -----------------------------------------------------------------------------
# 4. Inference Logic (SMILES -> 3D -> Prediction)
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_energy(model, scaler, smile, name="Molecule"):
    """
    Predicts the Internal Energy (U0) for a given SMILES string.
    
    Steps:
    1. Parse SMILES and Add Hydrogens.
    2. Generate 3D Conformer (ETKDGv3 + MMFF).
    3. Tensorize Inputs (z, pos).
    4. Model Prediction (Scaled).
    5. Inverse Transform to Real Unit.
    """
    print(f"\n[Analysis] Target: {name}")
    print(f"   -> SMILES: {smile}")

    # A. Molecule Preparation
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print("   [!] Error: Invalid SMILES string.")
        return None

    mol = Chem.AddHs(mol) # Crucial for 3D geometry

    # B. 3D Conformer Generation
    # QM9 has ground-truth 3D, but for inference on new molecules, 
    # we must estimate the 3D structure using RDKit.
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status == -1:
        print("   [!] Error: 3D Embedding failed.")
        return None

    # Force Field Optimization (Refining the geometry)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass # Ignore optimization failures, proceed with embedded coords

    # C. Feature Extraction
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    batch = torch.zeros(z.size(0), dtype=torch.long) # Single graph batch

    # D. Device Transfer
    pos = pos.to(device)
    z = z.to(device)
    batch = batch.to(device)

    # E. Prediction
    pred_scaled = model(z, pos, batch)
    
    # F. Inverse Scaling (Convert Z-Score back to Real Value)
    # The scaler expects a 2D array [n_samples, n_features]
    pred_real = scaler.inverse_transform(pred_scaled.cpu().numpy())
    
    value = pred_real[0][0]
    print(f"   -> Predicted Internal Energy (U0): {value:.4f}")
    return value


# -----------------------------------------------------------------------------
# 5. Execution Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== SchNet QM9 Inference Interface ===")
    
    try:
        # Load Resources
        model, scaler = load_system()
        
        # Inference Case Studies
        # Note: Values are abstract internal energies (likely Hartree or eV depending on QM9 source)
        
        # 1. Aspirin
        predict_energy(
            model, scaler, 
            smile="CC(=O)OC1=CC=CC=C1C(=O)O", 
            name="Aspirin"
        )
        
        # 2. Tylenol (Acetaminophen)
        predict_energy(
            model, scaler, 
            smile="CC(=O)NC1=CC=C(C=C1)O", 
            name="Tylenol"
        )
        
        # 3. Benzene (Simple Benchmark)
        predict_energy(
            model, scaler, 
            smile="C1=CC=CC=C1", 
            name="Benzene"
        )

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")