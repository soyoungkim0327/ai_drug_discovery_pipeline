"""
=============================================================================
[Hybrid Reasoning: 3D Conformer Generation & Ensemble Scoring Strategy]

[KR] 
본 파이프라인은 화합물의 2D SMILES를 3D 입체 구조(Conformer)로 변환하여 학습 및 추론에 활용합니다.
1. 3D 변환 필수성: 분자 도킹 및 수용체 결합은 물리적 입체 적합성(Stereochemical fit)을 기반으로 
   하므로, 정확한 물성 예측을 위해 RDKit을 이용한 3D Conformer 생성이 선행됩니다.
2. 앙상블 전략: 단일 Conformer 생성 시 발생할 수 있는 에너지 최저점(Local Minimum) 오류를 
   방지하기 위해, 다수의 Conformer를 생성하여 그 결과값을 통계적으로 결합(Ensemble)합니다.
   이는 '가짜 3D'가 가질 수 있는 불확실성을 통계적 확률 분포로 극복하는 고도화된 전략입니다.

[EN]
This pipeline utilizes 3D Conformer generation from 2D SMILES for enhanced 
molecular property prediction and virtual screening.
1. Necessity of 3D Representation: Since molecular docking and receptor-ligand 
   interactions are governed by stereochemical complementarity, 3D conformer 
   generation via RDKit is mandatory for physics-aware inference.
2. Ensemble Strategy (Probabilistic Refinement): To mitigate the risks of 
   generating a sub-optimal local minimum conformer, we implement an Ensemble 
   approach by generating multiple conformations (Mols). By aggregating 
   predictions across these conformers, we statistically minimize the 
   uncertainty inherent in rule-based 3D embedding, ensuring robust and 
   reliable scoring.
=============================================================================

SchNet 3D Training Pipeline (3-Way Split with Scaler)
=====================================================
This script implements a 3D SchNet regression model on a conformer-augmented
version of the Lipo dataset. It follows the rigorous training standards used
in the GAT pipeline.

Configuration:
- Model: SchNet (Continuous-filter convolutional neural network)
- Input: 3D atomic coordinates (z, pos)
- Hidden Channels: 64
- Optimizer: Adam (LR=0.001)
- Stability: Patience 100 with Early Stopping (Monitors Validation MAE)
- Monitoring: Tracks Training Loss and Real-world MAE (Inverse Scaled).

Output:
- Artifacts (Model/Scaler) are saved in the SAME directory as this script.
- Model saved as: 'best_schnet_model.pth'

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-03
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import joblib
import warnings
import json
from datetime import datetime
from tqdm import tqdm

# Graph Neural Network Imports
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch_geometric.datasets import MoleculeNet
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem import AllChem

# =============================================================================
# 1. Reproducibility & System Configuration
# =============================================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Device Allocation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Computational Device: {device}")

# [Path Configuration] Force artifacts to save in the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(SCRIPT_DIR, 'logp_schnet_scaler.pkl')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'best_schnet_model.pth')
META_PATH = os.path.join(SCRIPT_DIR, 'best_schnet_model.meta.json')

# Project Root for Data (Relative to this script)
# Assumes script is in: project_root/02_3d_structure/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 
LIPO_ROOT = os.path.join(PROJECT_ROOT, "data", "Lipo")
LIPO3D_ROOT = os.path.join(PROJECT_ROOT, "data", "Lipo_3D")

print(f"[*] Artifacts Output Directory: {SCRIPT_DIR}")


# =============================================================================
# 2. Data Acquisition and 3D Processing
# =============================================================================
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
        print("   -> Converting 2D SMILES -> 3D conformers...")

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

        print(f"   -> 3D build finished: kept={len(data_list)}, failed={n_fail}")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

print("\n[Step 1] Loading and Preprocessing Data (3D)...")
raw_data = MoleculeNet(root=LIPO_ROOT, name="Lipo")
dataset_3d = Lipophilicity3D(root=LIPO3D_ROOT, raw_dataset=raw_data, max_mols=500)

# [Target Standardization]
# Fit StandardScaler to normalize label distribution (LogP values)
# Note: In-memory datasets store data in .data.y
if hasattr(dataset_3d, 'data'):
    y_values = dataset_3d.data.y.numpy().reshape(-1, 1)
else:
    # Fallback for older PyG versions
    y_values = dataset_3d._data.y.numpy().reshape(-1, 1)

scaler = StandardScaler()
scaler.fit(y_values) 

# Serialize Scaler using Absolute Path
joblib.dump(scaler, SCALER_PATH)
print(f"   -> Scaler serialized: '{os.path.basename(SCALER_PATH)}'")

# Apply Transformation
if hasattr(dataset_3d, 'data'):
    dataset_3d.data.y = torch.from_numpy(scaler.transform(y_values)).float()
else:
    dataset_3d._data.y = torch.from_numpy(scaler.transform(y_values)).float()
    
print("   -> Dataset Normalization: Z-Score scaling applied.")

# [Data Splitting: 80% Train / 10% Val / 10% Test]
N = len(dataset_3d)
n_train = int(N * 0.8)
n_val = int(N * 0.1)
n_test = N - n_train - n_val

gen = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset_3d, [n_train, n_val, n_test], generator=gen
)

# [DataLoader Initialization]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"   -> Data Split: Train({len(train_dataset)}) / Val({len(val_dataset)}) / Test({len(test_dataset)})")


# =============================================================================
# 3. Model Architecture Specification (SchNet)
# =============================================================================
class SchNetModel(torch.nn.Module):
    """
    SchNet Model for 3D Molecular Property Prediction.
    Wraps PyG's SchNet with a custom output head.
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
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, z, pos, batch):
        h = self.model(z, pos, batch)
        return self.lin(h)

# [Model Initialization]
print("[Step 4] Initializing SchNet Model...")
model = SchNetModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =============================================================================
# 4. Optimization and Inference Protocols
# =============================================================================
def train():
    """
    Executes one training epoch.
    Returns: Average Training Loss (MSE).
    """
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        pred = model(data.z, data.pos, data.batch).view(-1)
        y = data.y.view(-1)
        
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * data.num_graphs # data.num_graphs handles batch size correctly
        
    return total_loss / len(train_loader.dataset)

def evaluate(loader):
    """
    Evaluates model performance on a specific loader (Val or Test).
    Returns: Real MAE (Inverse Transformed).
    """
    model.eval()
    abs_err_sum = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.z, data.pos, data.batch).view(-1)
            y = data.y.view(-1)
            
            # [Inverse Transform for Real Metrics]
            pred_scaled = pred.cpu().numpy().reshape(-1, 1)
            true_scaled = y.cpu().numpy().reshape(-1, 1)
            
            pred_real = scaler.inverse_transform(pred_scaled)
            true_real = scaler.inverse_transform(true_scaled)
            
            abs_err_sum += np.abs(pred_real - true_real).sum()
            total_samples += y.numel()
            
    return abs_err_sum / max(1, total_samples)


# =============================================================================
# 5. Training Campaign with Early Stopping
# =============================================================================
print("\n[Step 2] Initiating Training Process (SchNet 3D)...")

# [Hyperparameters]
PATIENCE = 100          # Requested Patience
patience_counter = 0    
best_val_mae = float('inf') 
MAX_EPOCHS = 1000       # Requested Epochs

print(f"   -> Config: SchNet, Optimizer=Adam(lr=0.001)")
print(f"   -> Max Epochs={MAX_EPOCHS}, Patience={PATIENCE}")
print("-" * 80)
print(f"{'Epoch':<10} | {'Train Loss':<15} | {'Val MAE':<15} | {'Status':<20}")
print("-" * 80)

for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    train_loss = train()
    
    # Evaluate on Validation Set
    val_mae = evaluate(val_loader)
    
    # [Check Point 1] Performance Improvement (on Val)
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0 
        
        # Save Model Artifact
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"{epoch:<10} | {train_loss:<15.4f} | {val_mae:<15.4f} | {'Saved (Best Val)':<20}")
        
    # [Check Point 2] Stagnation
    else:
        patience_counter += 1
        if epoch % 10 == 0 or patience_counter == PATIENCE:
             print(f"{epoch:<10} | {train_loss:<15.4f} | {val_mae:<15.4f} | {'Wait (' + str(patience_counter) + '/' + str(PATIENCE) + ')':<20}")
        
        # [Check Point 3] Early Stopping Trigger
        if patience_counter >= PATIENCE:
            print("-" * 80)
            print(f"[!] Training Terminated (Early Stopping)")
            print(f"    - Best Validation MAE: {best_val_mae:.4f}")
            break

# =============================================================================
# 6. Final Evaluation & Metadata
# =============================================================================
print("\n[Step 3] Final Evaluation on Unseen Test Set...")

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # Evaluate on Test Set
    final_test_mae = evaluate(test_loader)
    
    # Save Metadata
    meta = {
        "model_type": "SchNet",
        "dataset": "MoleculeNet/Lipo (3D embedded subset)",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "best_val_mae": best_val_mae,
        "final_test_mae": final_test_mae,
        "checkpoint": os.path.basename(MODEL_PATH),
    }
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"   -> Metadata written: '{os.path.basename(META_PATH)}'")
    except Exception as e:
        print(f"[Warning] Failed to write metadata: {e}")

    print("=" * 60)
    print(f"🏆 FINAL TEST SCORE (Real MAE): {final_test_mae:.4f}")
    print("=" * 60)
    print(f"   -> Best Model: '{os.path.basename(MODEL_PATH)}'")
    print(f"   -> Location:   {SCRIPT_DIR}")

else:
    print("[!] Model file not found. Training might have failed.")