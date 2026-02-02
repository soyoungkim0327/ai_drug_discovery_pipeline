from torch_geometric.datasets import MoleculeNet
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

ds = MoleculeNet(root="data/Lipo", name="Lipo")

def fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) if m else None

def topk_neighbors(query_smiles, k=5):
    qfp = fp(query_smiles)
    sims = []
    for i in range(len(ds)):
        smi = getattr(ds[i], "smiles", None)
        if not smi:
            continue
        dfp = fp(smi)
        if dfp is None:
            continue
        s = DataStructs.TanimotoSimilarity(qfp, dfp)
        sims.append((s, i, smi, float(ds[i].y)))
    sims.sort(reverse=True, key=lambda x: x[0])
    return sims[:k]

aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
tylenol = "CC(=O)NC1=CC=C(C=C1)O"

print("== aspirin neighbors ==")
for s,i,smi,y in topk_neighbors(aspirin, k=5):
    print(f"sim={s:.3f} idx={i} y={y:.3f} smiles={smi}")

print("\n== tylenol neighbors ==")
for s,i,smi,y in topk_neighbors(tylenol, k=5):
    print(f"sim={s:.3f} idx={i} y={y:.3f} smiles={smi}")
