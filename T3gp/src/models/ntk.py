from __future__ import annotations

"""
ntk.py

Utilities for computing empirical Neural Tangent Kernels (NTKs) for the model used in
nn_train.py, and using the NTK as a Gaussian Process (GP) / kernel ridge regression
(KRR) predictor.

This module is designed to be called in two explicit regimes:
  1) "init":   linearize at random initialization (no training)
  2) "post":   linearize at a trained model after optimization

Core public entrypoints:
  - ntk_gp_predict(...): GP posterior mean/variance for f(x) from linearized model
  - run_ntk_stage(...): stage-aware wrapper used by nn_train.py
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.func import functional_call, jacrev
from models.nn_models import MLPFModel


# ----------------------
# Basic helpers
# ----------------------

def params_and_buffers(model: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return parameter/buffer pytrees for torch.func.functional_call."""
    return dict(model.named_parameters()), dict(model.named_buffers())


def scalar_f_from_out(out_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Extract a scalar f(x) on the grid from the model's output dict.

    Convention:
      - if out["f_grid"] has shape (N,2), interpret [:,0] as mean and [:,1] as logvar.
      - else interpret out["f_grid"] as the mean directly.
    """
    f = out_dict["f_grid"]
    if f.ndim == 2 and f.shape[1] == 2:
        f = f[:, 0]
    return f.reshape(-1)


def jacobian_y_pred(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    xgrid_torch: torch.Tensor,
    W_block: torch.Tensor,
) -> torch.Tensor:
    """
    Jacobian of y_pred = W @ f_grid(xgrid) w.r.t. parameters.

    Returns:
        J_y: (Ndat, P)
    """

    def ypred_from_params(p):
        out = functional_call(model, (p, buffers), (xgrid_torch,))
        f = scalar_f_from_out(out)      # (Ngrid,)
        y = W_block @ f                 # (Ndat,)
        return y

    Jtree = jacrev(ypred_from_params)(params)  # leaves: (Ndat, *param_shape)
    J = torch.cat([leaf.reshape(W_block.shape[0], -1) for leaf in Jtree.values()], dim=1)
    return J


def jacobian_f_mu(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    x_pred_torch: torch.Tensor,
) -> torch.Tensor:
    """
    Jacobian of f_grid(x_pred) w.r.t. parameters.

    Returns:
        J_f: (Npred, P)
    """

    def f_from_params(p):
        out = functional_call(model, (p, buffers), (x_pred_torch,))
        return scalar_f_from_out(out)   # (Npred,)

    Jtree = jacrev(f_from_params)(params)
    n = x_pred_torch.shape[0]
    J = torch.cat([leaf.reshape(n, -1) for leaf in Jtree.values()], dim=1)
    return J


def _subsample_rows(
    n: int,
    max_n: Optional[int],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return an index tensor selecting up to max_n rows, or None for no subsample."""
    if max_n is None or n <= max_n:
        return None
    return torch.randperm(n, device=device)[:max_n]


def _subsample_pred_linspace(
    n: int,
    max_n: Optional[int],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Subsample prediction points using linspace indices (deterministic)."""
    if max_n is None or n <= max_n:
        return None
    return torch.linspace(0, n - 1, steps=max_n, device=device).long()


# ----------------------
# NTK as GP / KRR
# ----------------------

@torch.no_grad()
def ntk_gp_predict(
    model: torch.nn.Module,
    *,
    xgrid_torch: torch.Tensor,        # (Ngrid,1) grid used inside model to produce f_grid
    W_train: torch.Tensor,            # (Ntr,Ngrid)
    y_train: torch.Tensor,            # (Ntr,)
    x_pred_torch: torch.Tensor,       # (Npred,1)
    noise_train: torch.Tensor,        # (Ntr,Ntr) observation noise/covariance in y-space
    ridge: float = 1e-6,              # extra diagonal ridge for numerical stability
    max_train_points: Optional[int] = None,
    max_pred_points: Optional[int] = None,
    return_full_cov: bool = True,
) -> Dict[str, Any]:
    """
    Compute GP posterior over f(x_pred) using the empirical NTK linearization of `model`.

    Model:
        y = W f(xgrid)   (data space)
        f is parameterized by the NN evaluated on xgrid_torch.

    Linearization at current model parameters θ:
        y(θ) ≈ y0 + J_y δθ
        f(θ) ≈ f0 + J_f δθ

    Induced kernels:
        K_yy = J_y J_y^T
        K_yf = J_y J_f^T
        diag(K_ff) = rowwise ||J_f||^2

    Posterior (ridge-stabilized):
        A = K_yy + noise_train + ridge I
        α = A^{-1}(y - y0)
        f_mean = f0 + K_fy α
        var diag = diag(K_ff - K_fy A^{-1} K_yf)

    Returns dict with tensors:
        f_mean, f_var, train_sub_idx, pred_sub_idx
    """
    device = xgrid_torch.device
    dtype = xgrid_torch.dtype
    # print("in ntk predict gp")
    # print("return_full_cov: ", return_full_cov)

    ntr_full = W_train.shape[0]
    idx_tr = _subsample_rows(ntr_full, max_train_points, device=device)
    if idx_tr is not None:
        W_ntk = W_train[idx_tr]
        y_ntk = y_train[idx_tr]
        noise_ntk = noise_train[idx_tr][:, idx_tr]
    else:
        W_ntk = W_train
        y_ntk = y_train
        noise_ntk = noise_train

    npred_full = x_pred_torch.shape[0]
    idxp = _subsample_pred_linspace(npred_full, max_pred_points, device=device)
    x_pred_used = x_pred_torch[idxp] if idxp is not None else x_pred_torch

    model.eval()
    params_ntk, buffers_ntk = params_and_buffers(model)

    # Baseline means
    out_train = functional_call(model, (params_ntk, buffers_ntk), (xgrid_torch,))
    f0_train = scalar_f_from_out(out_train)           # (Ngrid,)
    y0 = W_ntk @ f0_train                             # (Nntk,)

    out_pred = functional_call(model, (params_ntk, buffers_ntk), (x_pred_used,))
    f0_pred = scalar_f_from_out(out_pred)             # (Npred,)

    # Jacobians + kernels
    J_y = jacobian_y_pred(model, params_ntk, buffers_ntk, xgrid_torch, W_ntk)   # (Nntk,P)
    J_f = jacobian_f_mu(model, params_ntk, buffers_ntk, x_pred_used)           # (Npred,P)

    K_yy = J_y @ J_y.T                                  # (Nntk,Nntk)
    K_yf = J_y @ J_f.T                                  # (Nntk,Npred)
    K_fy = K_yf.T                                       # (Npred,Nntk)
    diag_Kff = (J_f * J_f).sum(dim=1)                   # (Npred,)

    Nntk = K_yy.shape[0]
    A = K_yy + noise_ntk + float(ridge) * torch.eye(Nntk, device=device, dtype=dtype)

    # Cholesky solves (stable)
    L = torch.linalg.cholesky(A)

    rhs = (y_ntk - y0).reshape(-1, 1)                   # (Nntk,1)
    alpha = torch.cholesky_solve(rhs, L)                # (Nntk,1)

    f_mean = f0_pred + (K_fy @ alpha).reshape(-1)       # (Npred,)

    sol = torch.cholesky_solve(K_yf, L)                 # (Nntk,Npred)
    diag_cond = (K_yf * sol).sum(dim=0)                 # (Npred,)
    f_var = (diag_Kff - diag_cond).clamp_min(0.0)       # (Npred,)

    out = {
        "f_mean": f_mean,
        "f_var": f_var,
        "train_sub_idx": idx_tr,
        "pred_sub_idx": idxp,
    }

    if return_full_cov:
        # print("in return full cov")
        K_ff = J_f @ J_f.T                              # (Npred, Npred)
        K_cond = K_fy @ sol                             # (Npred, Npred) = K_fy A^{-1} K_yf
        f_cov = K_ff - K_cond                           # posterior cov
        f_cov = 0.5 * (f_cov + f_cov.T)                 # symmetrize for numerical safety
        out["f_cov"] = f_cov

    return out


def ntk_diagnostics(
    model: torch.nn.Module,
    *,
    xgrid_torch: torch.Tensor,
    W_train: torch.Tensor,
    max_train_points: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Cheap diagnostics for the empirical NTK in y-space:
        K_yy = J_y J_y^T

    Returns trace + eigenvalues (on the subsampled block if requested).
    """
    device = xgrid_torch.device

    ntr_full = W_train.shape[0]
    idx_tr = _subsample_rows(ntr_full, max_train_points, device=device)
    W_ntk = W_train[idx_tr] if idx_tr is not None else W_train

    model.eval()
    params_ntk, buffers_ntk = params_and_buffers(model)
    J_y = jacobian_y_pred(model, params_ntk, buffers_ntk, xgrid_torch, W_ntk)
    K_yy = J_y @ J_y.T

    eigs = torch.linalg.eigvalsh(K_yy).detach().cpu().numpy()
    return {
        "trace": float(torch.trace(K_yy).detach().cpu()),
        "eigs": eigs,
        "n_train_used": int(W_ntk.shape[0]),
        "train_sub_idx": idx_tr,
    }


def run_ntk_stage(
    *,
    stage: str,                       # "init" | "post"
    model: torch.nn.Module,
    xgrid_torch: torch.Tensor,
    xgrid_np: np.ndarray,
    W_train: torch.Tensor,
    y_train: torch.Tensor,
    C_train: Optional[torch.Tensor],
    x_pred_torch: torch.Tensor,
    x_pred_np: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience wrapper used by nn_train.py.

    Handles config knobs for:
      - enabling/disabling at explicit stages ("init" or "post")
      - choosing noise model: data covariance C vs lambda*I
      - ridge stabilization
      - subsampling train/pred points
      - returning mean/var + metadata
    """
    ntk_cfg = cfg.get("ntk", {})

    if not bool(ntk_cfg.get("gp_eval", False)):
        return {}

    stage = str(stage).lower()
    if stage not in {"init", "post"}:
        raise ValueError(f"run_ntk_stage only supports stage='init' or 'post', got {stage!r}.")

    when = str(ntk_cfg.get("when", "none")).lower()
    if when not in {"none", "init", "post"}:
        raise ValueError(f"ntk.when must be one of 'none', 'init', or 'post', got {when!r}.")

    if stage != when:
        return {}

    use_data_cov = bool(ntk_cfg.get("use_data_cov", True))
    ridge = float(ntk_cfg.get("ridge", 1e-6))
    lam = float(ntk_cfg.get("lambda", 1e-6))
    max_train_points = ntk_cfg.get("max_train_points", None)
    max_pred_points = ntk_cfg.get("max_pred_points", None)
    max_pred_points = int(max_pred_points) if max_pred_points is not None else None

    if use_data_cov:
        if C_train is None:
            raise ValueError("ntk.gp.use_data_cov=True but C_train is None.")
        noise = C_train
    else:
        noise = lam * torch.eye(W_train.shape[0], device=W_train.device, dtype=W_train.dtype)

    pred = ntk_gp_predict(
        model,
        xgrid_torch=xgrid_torch,
        W_train=W_train,
        y_train=y_train,
        x_pred_torch=x_pred_torch,
        noise_train=noise,
        ridge=ridge,
        max_train_points=max_train_points,
        max_pred_points=max_pred_points,
    )

    f_mean = pred["f_mean"].detach().cpu().numpy().astype(np.float64)
    f_var = pred["f_var"].detach().cpu().numpy().astype(np.float64)

    f_std = np.sqrt(np.maximum(f_var, 0.0))

    f_lo68 = f_mean - 1.0  * f_std
    f_hi68 = f_mean + 1.0  * f_std
    f_lo95 = f_mean - 1.96 * f_std
    f_hi95 = f_mean + 1.96 * f_std

    out: Dict[str, Any] = {
        f"{stage}_ntk_xgrid": x_pred_np,
        f"{stage}_ntk_f_mean": f_mean,
        f"{stage}_ntk_f_var": f_var,
        f"{stage}_ntk_f_std": f_std,
        f"{stage}_ntk_f_lo68": f_lo68,
        f"{stage}_ntk_f_hi68": f_hi68,
        f"{stage}_ntk_f_lo95": f_lo95,
        f"{stage}_ntk_f_hi95": f_hi95,
        f"{stage}_ntk_used_data_cov": bool(use_data_cov),
        f"{stage}_ntk_lambda": float(lam),
        f"{stage}_ntk_ridge": float(ridge),
    }
    if "f_cov" in pred and pred["f_cov"] is not None:
        out[f"{stage}_ntk_f_cov"] = pred["f_cov"].detach().cpu().numpy().astype(np.float64)
    if pred["train_sub_idx"] is not None:
        out[f"{stage}_ntk_train_sub_idx"] = pred["train_sub_idx"].detach().cpu().numpy()
    if pred["pred_sub_idx"] is not None:
        out[f"{stage}_ntk_pred_sub_idx"] = pred["pred_sub_idx"].detach().cpu().numpy()

    # Optional diagnostics
    if bool(ntk_cfg.get("diagnostics", False)):
        diag = ntk_diagnostics(
            model,
            xgrid_torch=xgrid_torch,
            W_train=W_train,
            max_train_points=max_train_points,
        )
        out[f"{stage}_ntk_trace"] = diag["trace"]
        out[f"{stage}_ntk_eigs"] = diag["eigs"]
        out[f"{stage}_ntk_n_train_used"] = diag["n_train_used"]

    return out

def run_ntk(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone NTK model mode.
    Called by your pipeline when cfg["model"]["type"] == "ntk".

    Returns NN-summary-style keys directly so downstream code can treat the
    standalone NTK route like a GP/kernel-regression run.
    """

    device = torch.device(cfg.get("nn", {}).get("device", "cpu"))
    dtype = torch.float32

    meta = ds.get("meta", {})
    print("NTK at init")

    # grid & operator
    xgrid = ds["xgrid"].astype(np.float32)                 # (Ngrid,)
    W = ds["W"].astype(np.float32)                         # (Ndat, Ngrid)
    y = ds["y"].astype(np.float32) 
    xt3_true = np.asarray(meta.get("xt3_true", []), float).ravel()                        # (Ndat,)

    # data covariance optional
    C_np = ds.get("C", None)
    C = None if C_np is None else C_np.astype(np.float32)

    # prediction grid: prefer xgrid_ext if present
    xgrid_ext = meta.get("xgrid_ext", None)
    if xgrid_ext is not None and len(xgrid_ext) > 0:
        x_pred_np = np.asarray(xgrid_ext, dtype=np.float64).ravel()
        x_pred_torch = torch.tensor(x_pred_np.astype(np.float32), device=device, dtype=dtype).unsqueeze(1)
    else:
        x_pred_np = xgrid.astype(np.float64)
        x_pred_torch = torch.tensor(xgrid, device=device, dtype=dtype).unsqueeze(1)

    # tensors
    xgrid_torch = torch.tensor(xgrid, device=device, dtype=dtype).unsqueeze(1)
    W_torch = torch.tensor(W, device=device, dtype=dtype)
    y_torch = torch.tensor(y, device=device, dtype=dtype).reshape(-1)

    C_torch = None
    if C is not None:
        C_torch = torch.tensor(C, device=device, dtype=dtype)

    # build model (same as training uses; no training here)
    nncfg = cfg.get("nn", {})
    torch.manual_seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    model = MLPFModel(
        hidden=nncfg.get("hidden", [64, 64]),
        activation=str(nncfg.get("activation", "tanh")),
        dropout=float(nncfg.get("dropout", 0.0)),
        out_dim=int(nncfg.get("out_dim", 1)),
        scaling=nncfg.get("scaling", True),
        init_alpha=float(nncfg.get("init_alpha", 1.0)),
        init_beta=float(nncfg.get("init_beta", 3.0)),
        transforms=cfg.get("transforms", {}),
    ).to(device=device, dtype=dtype)

    theta0 = {n: p.detach().clone() for n, p in model.named_parameters()}

    model.eval()

    # decide noise model
    # print('in ntk')
    ntk_cfg = cfg.get("ntk", {})
    
    use_data_cov = bool(ntk_cfg.get("use_data_cov", True))
    ridge = float(ntk_cfg.get("ridge", 1e-6))
    lam = float(ntk_cfg.get("lambda", 1e-6))
    lambda_sr = float(cfg.get("loss", {}).get("lambda_sr", 0.0))
    print("lambda_sr: ", lambda_sr)
    t3_ref_int = float(np.trapz(xt3_true / xgrid, xgrid)) if xt3_true.size > 0 else None

    sumrule_added = False
    sigma2_sr = None

    if not bool(ntk_cfg.get("gp_eval", False)):
        return {}

    if lambda_sr > 0.0 and (t3_ref_int is not None):
        print("Using NTK init sum rules")
        print("lambda: ", lambda_sr)

        # x must be (Ngrid,), not (Ngrid,1)
        x = xgrid_torch.squeeze(1)                     # (Ngrid,)
        dx = x[1:] - x[:-1]                            # (Ngrid-1,)

        w = torch.zeros_like(x)                        # (Ngrid,)
        w[0] = 0.5 * dx[0] / x[0]
        w[-1] = 0.5 * dx[-1] / x[-1]
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:]) / x[1:-1]   # trapezoid weights for ∫ f/x dx

        # Append pseudo-observation: y_sr = ref, W_sr = w^T
        W_torch = torch.cat([W_torch, w[None, :]], dim=0)  # (Ndat+1, Ngrid)
        y_torch = torch.cat(
            [y_torch, torch.as_tensor([t3_ref_int], device=y_torch.device, dtype=y_torch.dtype)],
            dim=0,
        )                                                  # (Ndat+1,)

        # NN penalty lambda_sr*(I-ref)^2  ~  (I-ref)^2/(2*sigma^2)
        sigma2_sr = 1.0 / (2.0 * lambda_sr)
        sumrule_added = True

        # If using data covariance as noise, extend it to (Ndat+1,Ndat+1)
        if C_torch is not None:
            C_torch = torch.block_diag(
                C_torch,
                torch.as_tensor([[sigma2_sr]], device=C_torch.device, dtype=C_torch.dtype),
            )
    
    if use_data_cov:
        if C_torch is None:
            raise ValueError("ntk.use_data_cov=True but C_train is None.")
        noise = C_torch
    else:
        n = W_torch.shape[0]  # IMPORTANT: updated if sumrule_added
        noise = lam * torch.eye(n, device=W_torch.device, dtype=W_torch.dtype)
        if sumrule_added and (sigma2_sr is not None):
            noise[-1, -1] = sigma2_sr
            

    # optional subsampling knobs
    max_train_points = ntk_cfg.get("max_train_points", None)
    max_train_points = int(max_train_points) if max_train_points is not None else None
    max_pred_points = ntk_cfg.get("max_pred_points", None)
    max_pred_points = int(max_pred_points) if max_pred_points is not None else None

    # run NTK-GP
    pred = ntk_gp_predict(
        model,
        xgrid_torch=xgrid_torch,
        W_train=W_torch,
        y_train=y_torch,
        x_pred_torch=x_pred_torch,
        noise_train=noise,
        ridge=ridge,
        max_train_points=max_train_points,
        max_pred_points=max_pred_points,
    )

    drifts = []
    for n, p in model.named_parameters():
        drifts.append((p - theta0[n]).norm().item() / (theta0[n].norm().item() + 1e-12))
    print("median rel drift:", np.median(drifts), "max:", np.max(drifts))

    f_mean = pred["f_mean"].detach().cpu().numpy().astype(np.float64)
    f_var = pred["f_var"].detach().cpu().numpy().astype(np.float64)

    # store bands too (optional but useful)
    f_std = np.sqrt(np.maximum(f_var, 0.0))
    f_lo68 = f_mean - 1.0  * f_std
    f_hi68 = f_mean + 1.0  * f_std
    f_lo95 = f_mean - 1.96 * f_std
    f_hi95 = f_mean + 1.96 * f_std
    f_cov = None
    if "f_cov" in pred and pred["f_cov"] is not None:
        f_cov = pred["f_cov"].detach().cpu().numpy().astype(np.float64)
        f_cov = 0.5 * (f_cov + f_cov.T)
    else:
        f_cov = np.diag(f_var)

    out = {
        "xgrid": x_pred_np.astype(np.float64),
        "mean_curve": f_mean,
        "var_ens": f_var,
        "var_het": np.zeros_like(f_var),
        "lo68": f_lo68,
        "hi68": f_hi68,
        "lo95": f_lo95,
        "hi95": f_hi95,
        "cov_ens_f": f_cov,
        "cov_het_f": np.zeros_like(f_cov),
        "cov_tot_f": f_cov.copy(),
        "sigma_f": f_std,
        "xgrid_cov_ens_f": x_pred_np.astype(np.float64),
        # Backward-compatible aliases used elsewhere in the codebase
        "f_grid_mean": f_mean,
        "f_grid_var": f_var,
        "f_grid_std": f_std,
        "f_grid_lo68": f_lo68,
        "f_grid_hi68": f_hi68,
        "f_grid_lo95": f_lo95,
        "f_grid_hi95": f_hi95,
        "f_cov": f_cov,
        "x_pred_used": x_pred_np.astype(np.float64),
        "ntk_used_data_cov": bool(use_data_cov),
        "ntk_lambda": float(lam),
        "ntk_ridge": float(ridge),
    }

    return out
