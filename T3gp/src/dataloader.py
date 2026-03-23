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

    meta = {k: d[k] for k in d.files if k not in {"xgrid", "W", "c_yy", target}}

    # sanity checks
    if W.shape[1] != xgrid.shape[0]:
        raise ValueError(f"W shape {W.shape} incompatible with xgrid {xgrid.shape}")
    if W.shape[0] != y.shape[0]:
        raise ValueError(f"W rows {W.shape[0]} != len(y) {y.shape[0]}")
    if C.shape != (y.shape[0], y.shape[0]):
        raise ValueError(f"C shape {C.shape} != {(y.shape[0], y.shape[0])}")

    res = cfg.get("rescale", {}).get("enabled", False)
    if res:
        eps = float(cfg.get("rescale", {}).get("eps", 1e-12))
        s = float(np.std(y))
        s = max(s, eps)
        y = y / s
        C = C / (s**2)
        W = W / s
    else:
        s = 1.0

    # Return with names that NUTS code expects: C and y
    return {"xgrid": xgrid, "W": W, "C": C, "y": y, "meta": meta, "scale_s": s}
