import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

D = np.load(BASE_DIR / "burgers.npz")

u0, uT = D["u0"].astype(np.float32) , D["uT"].astype(np.float32)
NTR, NTE = 1000, 100
FULL = u0.shape[1]

def with_grid(a, res):
    return a[:, :: FULL // res]
