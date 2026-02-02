"""
QM9 Dataset Downloader & Inspector
==================================
This script automates the retrieval and inspection of the QM9 dataset,
a benchmark dataset for 3D geometric deep learning in chemistry.

Functionality:
1. Downloads the ~134k molecule dataset to a local directory.
2. Preprocesses the data using PyTorch Geometric's standard pipeline.
3. Displays statistical summaries and structure of the first data sample.

Dataset:
- Source: Quantum Machines 9 (QM9)
- Content: ~134k stable small organic molecules with 19 geometric/energetic properties.
- Features: 3D coordinates (pos), Atomic numbers (z), Ground-truth properties (y).

Author: Soyoung Kim (Assisted by AI)
Date: 2026-02-03
"""

import os
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

# 1. Configuration: Dataset Storage Path
# Note: The raw and processed data (approx. 130k molecules) will be persistently stored here.
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'QM9')

def download_and_inspect_qm9():
    print(f"[*] Initiating QM9 dataset download to: {DATA_PATH}")
    print("    (Note: First-time execution may take 5-10 minutes for download and preprocessing.)")

    # 2. Dataset Initialization (Download & Load)
    # PyTorch Geometric automatically handles download, extraction, and preprocessing.
    dataset = QM9(root=DATA_PATH)

    print("\n" + "="*60)
    print(f"✅ QM9 Dataset Ready!")
    print(f"[*] Total Molecules: {len(dataset)} (approx. 134k)")
    print(f"[*] Number of Node Features: {dataset.num_features}")
    print(f"[*] Number of Target Properties: {dataset.num_classes} (Energy, Gap, etc.)")
    print("="*60)

    # 3. Data Inspection (Sample Verification)
    # Retrieve the first molecule (Index 0, typically Methane/CH4) for structural check.
    data = dataset[0] 
    
    print("\n[Sample Inspection: Molecule ID 0]")
    print(f"1. 3D Coordinates (pos): \n{data.pos[:5]} ... (Total Atoms: {data.pos.shape[0]})")
    print(f"2. Atomic Numbers (z):   {data.z} (1=H, 6=C, 7=N, 8=O...)")
    print(f"3. Target Properties (y): {data.y} (Physical properties like U0, HOMO/LUMO)")
    
    # 4. Usage Note for the Researcher
    print("\n💡 [Integration Note]")
    print("This 'dataset' object is fully compatible with GNN architectures (GAT, SchNet).")
    print(f"Processed data has been saved to: '{DATA_PATH}/processed/data.pt'")

if __name__ == "__main__":
    download_and_inspect_qm9()