import os
import pandas as pd
from rdkit import Chem

# =============================================================================
# Configuration: Dataset Path Resolution
# =============================================================================
# Define a list of potential file paths to ensure robustness across different 
# directory structures (e.g., local environment vs. standard MoleculeNet structure).
possible_paths = [
    "data/Lipo/lipo/raw/Lipophilicity.csv",  # User-specific directory structure
    "data/Lipo/raw/Lipophilicity.csv",       # Standard MoleculeNet structure
    "data/Lipo/processed/Lipophilicity.csv"  # Alternative processed data path
]

csv_path = None
for path in possible_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    print("🦃 Error: CSV file not found. Please verify the directory structure.")
    print(f"Attempted paths: {possible_paths}")
    exit()

print(f"🍗 Dataset successfully located! Loading from: {csv_path}")

# =============================================================================
# Data Ingestion
# =============================================================================
# Load the raw dataset into a Pandas DataFrame for inspection.
df = pd.read_csv(csv_path)

# =============================================================================
# Target Definition & Utility Function
# =============================================================================
# Define query molecules with their standard SMILES strings for validation.
queries = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "tylenol": "CC(=O)NC1=CC=C(C=C1)O"
}

def get_canonical_smiles(smiles):
    """
    Converts a given SMILES string to its canonical form using RDKit.
    
    Canonicalization ensures that different string representations of the 
    same molecule can be directly compared (e.g., 'CCO' == 'OCC').
    
    Args:
        smiles (str): Input SMILES string.
        
    Returns:
        str: Canonical SMILES string if valid, otherwise None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Return isomeric SMILES to preserve stereochemistry information if present
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    return None

print("\n🦃 Starting Dataset Validation / Molecule Search...")

# =============================================================================
# Execution: Search & Validation Loop
# =============================================================================
for name, query_smi in queries.items():
    # Convert query molecule to canonical form for accurate matching
    canon_query = get_canonical_smiles(query_smi)
    
    found = False
    
    # Iterate through the dataset to find a match
    # Note: This performs a linear search (O(N)) comparing canonical SMILES.
    for idx, row in df.iterrows():
        dataset_smi = row['smiles'] # Assumes column name is 'smiles'
        dataset_val = row['exp']    # Assumes column name is 'exp' (Experimental LogP)
        
        # Canonicalize the dataset molecule for comparison
        canon_dataset = get_canonical_smiles(dataset_smi)
        
        # Check for exact structural match
        if canon_query and canon_dataset and (canon_query == canon_dataset):
            print(f"🍗 Match Found: {name} (Index: {idx})")
            print(f"   - Canonical SMILES: {dataset_smi}")
            print(f"   - Ground Truth (LogP): {dataset_val}")
            found = True
            break # Stop search once the target is found
            
    if not found:
        print(f"🦃 {name} : Not found in the current dataset.")