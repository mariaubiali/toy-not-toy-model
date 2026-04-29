from __future__ import annotations
import os
import numpy as np
from typing import Any, Dict


def load_dataset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    path = cfg["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    target = cfg.get("target", "y_pseudo")
    d = np.load(path, allow_pickle=True)

    # xgrid = np.asarray(d["xgrid"], float)
    xgrid = np.asarray(d["xgrid"], float)
    W = np.asarray(d["W"], float)

    scaling_cov = float(cfg.get("scaling_cov", 1.0))
    if scaling_cov <= 0:
        raise ValueError(f"Invalid scaling_cov={scaling_cov}. Covariance scaling must be > 0.")
    if scaling_cov != 1.0:
        print(f"[INFO] Scaling CY by factor {scaling_cov}")
    C = np.asarray(d["c_yy"], float) * scaling_cov

    if target not in d:
        raise KeyError(f"target={target} not in npz keys: {list(d.keys())}")
    y = np.asarray(d[target], float)
    y = np.asarray(d[target], float)
    if y.ndim == 2:
        if y.shape[0] == W.shape[0]:
            pass
        elif y.shape[1] == W.shape[0]:
            print(f"[INFO] Transposing target {target} from {y.shape} " f"to {(y.shape[1], y.shape[0])}")
            y = y.T
        else:
            raise ValueError(f"2D target {target} has shape {y.shape}, but neither axis matches " f"Ndata={W.shape[0]}")
    n_data = y.shape[0]

    meta = {k: d[k] for k in d.files if k not in {"xgrid", "W", "c_yy", target}}

    # sanity checks
    if y.ndim not in (1, 2):
        raise ValueError(f"Target {target} must have shape (Ndata,) or (Ndata, Nreplicas), got {y.shape}")
    if W.shape[1] != xgrid.shape[0]:
        raise ValueError(f"W shape {W.shape} incompatible with xgrid {xgrid.shape}")
    if W.shape[0] != n_data:
        raise ValueError(f"W rows {W.shape[0]} != y rows {n_data}")
    if C.shape != (n_data, n_data):
        raise ValueError(f"C shape {C.shape} != {(n_data, n_data)}")
    
    res = cfg.get("rescale", {}).get("enabled", False)
    if res:
        eps = float(cfg.get("rescale", {}).get("eps", 1e-12))

        # For y_l2, use the global std over all data points and all replicas.
        # This keeps one common scale for the whole L2 ensemble.
        s = float(np.std(y))
        s = max(s, eps)
        y = y / s
        C = C / (s**2)
        W = W / s
    else:
        s = 1.0

    # Return with names that NUTS code expects: C and y
    return {
        "xgrid": xgrid,
        "W": W,
        "C": C,
        "y": y,
        "target": target,
        "meta": meta,
        "scale_s": s,
    }
