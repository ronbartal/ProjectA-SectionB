"""Quick integrity check for an .npz artifact (usage: python check_npz.py PATH)."""
import sys

import numpy as np

path = sys.argv[1]
z = np.load(path, allow_pickle=True)
print("OK", path, "keys:", list(z.keys()), "nnz:", z["data"].shape[0])
