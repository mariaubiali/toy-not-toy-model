from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.func import functional_call, jacrev

from models.nn_models import MLPFModel


# ----------------------
# Utility helper
# ----------------------

def _param_buffers(model: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    return dict(model.named_parameters()), dict(model.named_buffers())

def _scalar_f_from_out(out_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Match train_nn_forward convention:
    - if out_dim=2 -> out["f_grid"][:,0] is mean, [:,1] is logvar
    - else out["f_grid"] is the mean directly
    """
    f = out_dict["f_grid"]
    if f.ndim == 2 and f.shape[1] == 2:
        f = f[:, 0]
    return f.reshape(-1)

def _jacobian_y_pred(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    xgrid_torch: torch.Tensor,
    W_block: torch.Tensor,
) -> torch.Tensor:
    """
    Jacobian of y_pred = W @ f_grid(xgrid) w.r.t parameters.
    Returns J_y with shape (Ndat, P).
    """

    def ypred_from_params(p):
        out = functional_call(model, (p, buffers), (xgrid_torch,))
        f = _scalar_f_from_out(out)      # (Ngrid,)
        y = W_block @ f                  # (Ndat,)
        return y

    Jtree = jacrev(ypred_from_params)(params)  # pytree leaves: (Ndat, *param_shape)
    J = torch.cat([leaf.reshape(W_block.shape[0], -1) for leaf in Jtree.values()], dim=1)
    return J


def _jacobian_f_grid(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    xgrid_torch: torch.Tensor,
) -> torch.Tensor:
    """
    Jacobian of f_grid(xgrid) w.r.t parameters.
    Returns J_f with shape (Ngrid, P).
    """

    def f_from_params(p):
        out = functional_call(model, (p, buffers), (xgrid_torch,))
        return _scalar_f_from_out(out)   # (Ngrid,)

    Jtree = jacrev(f_from_params)(params)
    n = xgrid_torch.shape[0]
    J = torch.cat([leaf.reshape(n, -1) for leaf in Jtree.values()], dim=1)
    return J


def _krr_solve(K: torch.Tensor, rhs: torch.Tensor, lam: float) -> torch.Tensor:
    n = K.shape[0]
    return torch.linalg.solve(
        K + lam * torch.eye(n, device=K.device, dtype=K.dtype),
        rhs,
    )

# -------------------------------
# start of actual NTK magic
# access from init baseline here
# -------------------------------

def run_ntk(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Init-time NTK baseline for setup y = W f(xgrid).

    Compute NTK at initialization in data-space:
      K_yy = J_y J_y^T  where y_pred = W f_theta(xgrid)

    Solve kernel ridge regression for residuals:
      alpha = (K_yy + lam I)^(-1) (y_tr - y0_tr)

    Map to function-space:
      f_pred = f0_pred + K_fy alpha, with K_fy = J_f J_y^T
    """

    device = torch.device(cfg.get("nn", {}).get("device", "cpu"))
    dtype = torch.float32

    meta = ds.get("meta", {})

    # use extrapolation xgrid if existing
    xgrid_ext = (
        meta.get("xgrid_ext").astype(np.float64)
        if "xgrid_ext" in meta
        else np.array([], dtype=np.float64)
    )

    xt3_true = np.asarray(meta.get("xt3_true", []), float).ravel()

    xgrid = ds["xgrid"].astype(np.float32)     # (Ngrid,)
    W = ds["W"].astype(np.float32)             # (Ndat, Ngrid)
    y = ds["y"].astype(np.float32)             # (Ndat,)

    n_grid = xgrid.shape[0]
    if xt3_true.size not in (0, n_grid):
        raise ValueError(
            f"xt3_true must be defined on xgrid: got {xt3_true.size} vs Ngrid={n_grid}"
        )

    x_torch = torch.tensor(xgrid, dtype=dtype, device=device).unsqueeze(1)  # (Ngrid,1)
    W_torch = torch.tensor(W, dtype=dtype, device=device)
    y_torch = torch.tensor(y, dtype=dtype, device=device).reshape(-1)

    # same split style as train_nn_forward
    nncfg = cfg.get("nn", {})
    seed = int(cfg.get("seed", 0))
    replica = int(cfg.get("replica", 0))  # keep compatibility
    val_frac = float(nncfg.get("val_frac", 0.2))

    # reproducibility for model init + split
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx_all = np.arange(y_torch.shape[0])
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=val_frac,
        random_state=seed + replica,
        shuffle=True,
    )

    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    W_tr_full = W_torch[train_idx_t, :]
    y_tr_full = y_torch[train_idx_t]

    # Optional subsampling of training points for feasibility
    ntk_cfg = cfg.get("ntk", {})
    max_n = ntk_cfg.get("max_train_points", None)
    max_n = int(max_n) if max_n is not None else None
    if max_n is not None and W_tr_full.shape[0] > max_n:
        perm = torch.randperm(W_tr_full.shape[0], device=device)[:max_n]
        W_tr = W_tr_full[perm]
        y_tr = y_tr_full[perm]
        train_idx_used = train_idx[perm.detach().cpu().numpy()]
    else:
        W_tr = W_tr_full
        y_tr = y_tr_full
        train_idx_used = train_idx

    # rebulid NN architecture for NTK, but not train it
    # just for eval?
    hidden = nncfg.get("hidden", [64, 64])
    activation = str(nncfg.get("activation", "tanh"))
    dropout = float(nncfg.get("dropout", 0.0))
    out_dim = int(nncfg.get("out_dim", 1))
    use_preproc = bool(nncfg.get("use_preproc", True))
    init_alpha = float(nncfg.get("init_alpha", 1.0))
    init_beta = float(nncfg.get("init_beta", 3.0))
    transforms = cfg.get("transforms", {})

    model = MLPFModel(
        hidden=hidden,
        activation=activation,
        dropout=dropout,
        out_dim=out_dim,
        use_preproc=use_preproc,
        init_alpha=init_alpha,
        init_beta=init_beta,
        transforms=transforms,
    ).to(device)
    model.eval()

    params, buffers = _param_buffers(model)

    # Prediction grid = ext if present, else original grid
    if xgrid_ext.size > 0:
        x_pred_np = xgrid_ext.astype(np.float64)
        x_pred_torch = torch.tensor(xgrid_ext.astype(np.float32), dtype=dtype, device=device).unsqueeze(1)
    else:
        x_pred_np = xgrid.astype(np.float64)
        x_pred_torch = x_torch

    # f0 and y0 at init
    with torch.no_grad():
        f0_train = _scalar_f_from_out(model(x_torch))   # (Ngrid,)
        y0_tr = W_tr @ f0_train                         # (Ntr,)

        f0_pred = _scalar_f_from_out(model(x_pred_torch))  # (N*,)

    # Jacobians at init
    J_y = _jacobian_y_pred(model, params, buffers, x_torch, W_tr)        # (Ntr, P)
    J_f = _jacobian_f_grid(model, params, buffers, x_pred_torch)         # (N*, P)

    # Kernels
    K_yy = J_y @ J_y.T                         # (Ntr, Ntr)
    K_fy = J_f @ J_y.T                         # (N*, Ntr)

    lam = float(ntk_cfg.get("lambda", 1e-6)) # what does Lambda do here?

    alpha = _krr_solve(K_yy, (y_tr - y0_tr), lam=lam)  # (Ntr,)
    f_pred = f0_pred + (K_fy @ alpha)                  # (N*,)

    res = {
        "xgrid": x_pred_np,
        "f_grid_mean": f_pred.detach().cpu().numpy().astype(np.float64),
        "loss_history": np.array([], dtype=np.float64),
        "train_idx": train_idx_used,
        "val_idx": val_idx,
        "ntk_init_lambda": lam,
        "ntk_init_n_train_used": int(W_tr.shape[0]),
    }
    return res