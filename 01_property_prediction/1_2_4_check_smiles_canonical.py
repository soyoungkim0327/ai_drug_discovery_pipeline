from rdkit import Chem

def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None

print(canon("CC(=O)NC1=CC=C(C=C1)O"))  # -> CC(=O)Nc1ccc(O)cc1
