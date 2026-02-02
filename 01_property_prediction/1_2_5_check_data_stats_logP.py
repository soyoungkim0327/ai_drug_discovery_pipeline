from torch_geometric.datasets import MoleculeNet
import numpy as np

ds = MoleculeNet(root="data/Lipo", name="Lipo")
y = ds.data.y.view(-1).cpu().numpy()
print("y_min=", float(np.min(y)), "y_max=", float(np.max(y)), "y_mean=", float(np.mean(y)), "y_std=", float(np.std(y)))
