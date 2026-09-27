import os
import torch
import numpy as np
import random

# Global configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(x):
    """Converts input to a float tensor on the configured device."""
    if isinstance(x, torch.Tensor):
        return x.to(DEVICE).float()
    return torch.tensor(x, dtype=torch.float32).to(DEVICE)

def to_numpy(x):
    """Converts tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)



# --- 1. Data Split Logic (From preprocess.ipynb) ---
def data_split_syn(X, y, n_cal_prop=0.01, n_train_prop=0.02, seed=0):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_ext   = int(0.20 * n)
    n_syn   = int(0.30 * n)
    n_mean  = int(n_train_prop * n)
    n_cal = int(n_cal_prop * n) # Note: Logic in notebook used 10% here, effectively splitting 5/5 later inside run_rcp_syn

    i = 0
    id_ext  = idx[i : i + n_ext];   i += n_ext
    id_syn  = idx[i : i + n_syn];   i += n_syn
    id_mean = idx[i : i + n_mean];  i += n_mean
    id_cal  = idx[i : i + n_cal];   i += n_cal
    id_test = idx[i : ]

    return {
        "ext":  (X[id_ext],  y[id_ext]),
        "syn":  (X[id_syn],  y[id_syn]), 
        "mean": (X[id_mean], y[id_mean]),
        "cal":  (X[id_cal],  y[id_cal]),
        "test": (X[id_test], y[id_test]),
    }
