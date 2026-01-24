from torch_geometric.datasets import MoleculeNet
ds = MoleculeNet(root="data/Lipo", name="Lipo")

queries = {
  "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "tylenol": "CC(=O)NC1=CC=C(C=C1)O"
}

for name, smi in queries.items():
    hit = [i for i in range(len(ds)) if getattr(ds[i], "smiles", None) == smi]
    if hit:
        i = hit[0]
        print(name, "FOUND y=", float(ds[i].y), "idx=", i)
    else:
        print(name, "not found")
