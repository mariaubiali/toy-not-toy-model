from __future__ import annotations

"""
ntk.py

Empirical and analytic NTK utilities.

Two distinct use cases live here:

1) Staged empirical NTK around a finite PyTorch model
   - used by nn_train.py via run_ntk_stage(...)
   - supports explicit stages "init" and "post"

2) Standalone NTK route for model.type == "ntk"
   - analytic infinite-width NTK GP / kernel regression when available
   - preserved finite-width empirical init implementation as a reference fallback
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
    return dict(model.named_parameters()), dict(model.named_buffers())


def scalar_f_from_out(out_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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
    def ypred_from_params(p):
        out = functional_call(model, (p, buffers), (xgrid_torch,))
        f = scalar_f_from_out(out)
        y = W_block @ f
        return y

    Jtree = jacrev(ypred_from_params)(params)
    J = torch.cat([leaf.reshape(W_block.shape[0], -1) for leaf in Jtree.values()], dim=1)
    return J


def jacobian_f_mu(
    model: torch.nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    x_pred_torch: torch.Tensor,
) -> torch.Tensor:
    def f_from_params(p):
        out = functional_call(model, (p, buffers), (x_pred_torch,))
        return scalar_f_from_out(out)

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
    if max_n is None or n <= max_n:
        return None
    return torch.randperm(n, device=device)[:max_n]


def _subsample_pred_linspace(
    n: int,
    max_n: Optional[int],
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if max_n is None or n <= max_n:
        return None
    return torch.linspace(0, n - 1, steps=max_n, device=device).long()


def _subsample_rows_np(n: int, max_n: Optional[int], seed: int) -> Optional[np.ndarray]:
    if max_n is None or n <= max_n:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def _subsample_pred_linspace_np(n: int, max_n: Optional[int]) -> Optional[np.ndarray]:
    if max_n is None or n <= max_n:
        return None
    return np.linspace(0, n - 1, num=max_n).round().astype(int)


def _symmetrize(C: np.ndarray) -> np.ndarray:
    return 0.5 * (C + C.T)


def _gp_posterior_np(
    K_yy: np.ndarray,
    K_yf: np.ndarray,
    K_ff: np.ndarray,
    y: np.ndarray,
    noise: np.ndarray,
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    K_yy = np.asarray(K_yy, dtype=np.float64)
    K_yf = np.asarray(K_yf, dtype=np.float64)
    K_ff = np.asarray(K_ff, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)

    n = K_yy.shape[0]
    A = _symmetrize(K_yy + noise + float(ridge) * np.eye(n, dtype=np.float64))
    L = np.linalg.cholesky(A)

    rhs = y.reshape(-1, 1)
    z = np.linalg.solve(L, rhs)
    alpha = np.linalg.solve(L.T, z)
    f_mean = (K_yf.T @ alpha).reshape(-1)

    v = np.linalg.solve(L, K_yf)
    f_cov = _symmetrize(K_ff - v.T @ v)
    f_var = np.clip(np.diag(f_cov), 0.0, None)
    f_std = np.sqrt(f_var)
    return f_mean, f_std, f_cov


def _trapz_sumrule_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    dx = x[1:] - x[:-1]
    w = np.zeros_like(x)
    w[0] = 0.5 * dx[0] / x[0]
    w[-1] = 0.5 * dx[-1] / x[-1]
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:]) / x[1:-1]
    return w


def _summary_from_gp_arrays(
    *,
    x_pred: np.ndarray,
    f_mean: np.ndarray,
    f_cov: np.ndarray,
    use_data_cov: bool,
    lam: float,
    ridge: float,
    mode: str,
) -> Dict[str, Any]:
    f_mean = np.asarray(f_mean, dtype=np.float64).reshape(-1)
    f_cov = _symmetrize(np.asarray(f_cov, dtype=np.float64))
    f_var = np.clip(np.diag(f_cov), 0.0, None)
    f_std = np.sqrt(f_var)
    f_lo68 = f_mean - 1.0 * f_std
    f_hi68 = f_mean + 1.0 * f_std
    f_lo95 = f_mean - 1.96 * f_std
    f_hi95 = f_mean + 1.96 * f_std

    return {
        "xgrid": np.asarray(x_pred, dtype=np.float64).reshape(-1),
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
        "xgrid_cov_ens_f": np.asarray(x_pred, dtype=np.float64).reshape(-1),
        # backward-compatible aliases used elsewhere
        "f_grid_mean": f_mean,
        "f_grid_var": f_var,
        "f_grid_std": f_std,
        "f_grid_lo68": f_lo68,
        "f_grid_hi68": f_hi68,
        "f_grid_lo95": f_lo95,
        "f_grid_hi95": f_hi95,
        "f_cov": f_cov,
        "x_pred_used": np.asarray(x_pred, dtype=np.float64).reshape(-1),
        "ntk_used_data_cov": bool(use_data_cov),
        "ntk_lambda": float(lam),
        "ntk_ridge": float(ridge),
        "ntk_mode": str(mode),
    }


# ----------------------
# NTK as GP / KRR (empirical finite-width around a PyTorch model)
# ----------------------

@torch.no_grad()
def ntk_gp_predict(
    model: torch.nn.Module,
    *,
    xgrid_torch: torch.Tensor,
    W_train: torch.Tensor,
    y_train: torch.Tensor,
    x_pred_torch: torch.Tensor,
    noise_train: torch.Tensor,
    ridge: float = 1e-6,
    max_train_points: Optional[int] = None,
    max_pred_points: Optional[int] = None,
    return_full_cov: bool = True,
) -> Dict[str, Any]:
    device = xgrid_torch.device
    dtype = xgrid_torch.dtype

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

    out_train = functional_call(model, (params_ntk, buffers_ntk), (xgrid_torch,))
    f0_train = scalar_f_from_out(out_train)
    y0 = W_ntk @ f0_train

    out_pred = functional_call(model, (params_ntk, buffers_ntk), (x_pred_used,))
    f0_pred = scalar_f_from_out(out_pred)

    J_y = jacobian_y_pred(model, params_ntk, buffers_ntk, xgrid_torch, W_ntk)
    J_f = jacobian_f_mu(model, params_ntk, buffers_ntk, x_pred_used)

    K_yy = J_y @ J_y.T
    K_yf = J_y @ J_f.T
    K_fy = K_yf.T
    diag_Kff = (J_f * J_f).sum(dim=1)

    nntk = K_yy.shape[0]
    A = K_yy + noise_ntk + float(ridge) * torch.eye(nntk, device=device, dtype=dtype)
    L = torch.linalg.cholesky(A)

    rhs = (y_ntk - y0).reshape(-1, 1)
    alpha = torch.cholesky_solve(rhs, L)
    f_mean = f0_pred + (K_fy @ alpha).reshape(-1)

    sol = torch.cholesky_solve(K_yf, L)
    diag_cond = (K_yf * sol).sum(dim=0)
    f_var = (diag_Kff - diag_cond).clamp_min(0.0)

    out = {
        "f_mean": f_mean,
        "f_var": f_var,
        "train_sub_idx": idx_tr,
        "pred_sub_idx": idxp,
    }

    if return_full_cov:
        K_ff = J_f @ J_f.T
        K_cond = K_fy @ sol
        f_cov = 0.5 * (K_ff - K_cond + (K_ff - K_cond).T)
        out["f_cov"] = f_cov

    return out


def ntk_diagnostics(
    model: torch.nn.Module,
    *,
    xgrid_torch: torch.Tensor,
    W_train: torch.Tensor,
    max_train_points: Optional[int] = None,
) -> Dict[str, Any]:
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
    stage: str,
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
    ntk_cfg = cfg.get("ntk", {})
    when = str(ntk_cfg.get("when", "none")).lower()
    if when not in {"none", "init", "post"}:
        raise ValueError(f"ntk.when must be one of 'none', 'init', 'post'; got {when!r}")
    if stage not in {"init", "post"}:
        raise ValueError(f"run_ntk_stage stage must be 'init' or 'post'; got {stage!r}")
    if not bool(ntk_cfg.get("gp_eval", False)):
        return {}
    if when != stage:
        return {}

    use_data_cov = bool(ntk_cfg.get("use_data_cov", True))
    ridge = float(ntk_cfg.get("ridge", 1e-6))
    lam = float(ntk_cfg.get("lambda", 1e-6))
    max_train_points = ntk_cfg.get("max_train_points", None)
    max_train_points = int(max_train_points) if max_train_points is not None else None
    max_pred_points = ntk_cfg.get("max_pred_points", None)
    max_pred_points = int(max_pred_points) if max_pred_points is not None else None

    if use_data_cov:
        if C_train is None:
            raise ValueError("ntk.use_data_cov=True but C_train is None.")
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
    f_lo68 = f_mean - 1.0 * f_std
    f_hi68 = f_mean + 1.0 * f_std
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


# ----------------------
# Standalone NTK route for model.type == "ntk"
# ----------------------

def _ntk_prefactor_scaling(x: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """
    NN-style physical prefactor for the analytic init NTK:
        pre(x) = x^alpha (1 - x)^beta

    Uses init_alpha and init_beta from cfg["nn"].
    """
    nncfg = cfg.get("nn", {})
    alpha = float(nncfg.get("init_alpha", 1.0))
    beta = float(nncfg.get("init_beta", 1.0))

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x_clip = np.clip(x, 1e-12, 1.0 - 1e-12)

    return (x_clip ** alpha) * ((1.0 - x_clip) ** beta)

def _prepare_standalone_ntk_inputs(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    meta = ds.get("meta", {})
    xgrid = np.asarray(ds["xgrid"], dtype=np.float64).ravel()
    W = np.asarray(ds["W"], dtype=np.float64)
    y = np.asarray(ds["y"], dtype=np.float64).ravel()
    xt3_true = np.asarray(meta.get("xt3_true", []), dtype=np.float64).ravel()
    C = None if ds.get("C", None) is None else np.asarray(ds["C"], dtype=np.float64)

    xgrid_ext = meta.get("xgrid_ext", None)
    if xgrid_ext is not None and len(xgrid_ext) > 0:
        x_pred = np.asarray(xgrid_ext, dtype=np.float64).ravel()
    else:
        x_pred = xgrid.copy()

    ntk_cfg = cfg.get("ntk", {})
    use_data_cov = bool(ntk_cfg.get("use_data_cov", True))
    ridge = float(ntk_cfg.get("ridge", 1e-6))
    lam = float(ntk_cfg.get("lambda", 1e-6))
    lambda_sr = float(cfg.get("loss", {}).get("lambda_sr", 0.0))
    t3_ref_int = float(np.trapz(xt3_true / xgrid, xgrid)) if xt3_true.size > 0 else None

    sigma2_sr = None
    sumrule_added = False
    # apply_sumrule = bool(cfg.get("ntk", {}).get("apply_sumrule", False))
    # if apply_sumrule and lambda_sr > 0.0 and (t3_ref_int is not None):
    #     w = _trapz_sumrule_weights(xgrid)
    #     W = np.concatenate([W, w[None, :]], axis=0)
    #     y = np.concatenate([y, np.asarray([t3_ref_int], dtype=np.float64)], axis=0)
    #     sigma2_sr = 1.0 / (2.0 * lambda_sr)
    #     sumrule_added = True
    #     if C is not None:
    #         C_aug = np.zeros((C.shape[0] + 1, C.shape[1] + 1), dtype=np.float64)
    #         C_aug[:-1, :-1] = C
    #         C_aug[-1, -1] = sigma2_sr
    #         C = C_aug

    if use_data_cov:
        if C is None:
            raise ValueError("ntk.use_data_cov=True but dataset covariance C is missing.")
        noise = C
    else:
        noise = lam * np.eye(W.shape[0], dtype=np.float64)
        # if sumrule_added and (sigma2_sr is not None):
        #     noise[-1, -1] = sigma2_sr

    max_train_points = ntk_cfg.get("max_train_points", None)
    max_train_points = int(max_train_points) if max_train_points is not None else None
    max_pred_points = ntk_cfg.get("max_pred_points", None)
    max_pred_points = int(max_pred_points) if max_pred_points is not None else None

    return {
        "xgrid": xgrid,
        "x_pred": x_pred,
        "W": W,
        "y": y,
        "noise": noise,
        "use_data_cov": use_data_cov,
        "ridge": ridge,
        "lam": lam,
        "max_train_points": max_train_points,
        "max_pred_points": max_pred_points,
        "seed": int(cfg.get("seed", 0)),
    }


def run_ntk_empirical_ref(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Preserved old standalone init implementation using a finite PyTorch model."""
    if not bool(cfg.get("ntk", {}).get("gp_eval", False)):
        return {}

    prepared = _prepare_standalone_ntk_inputs(ds, cfg)
    xgrid = prepared["xgrid"]
    x_pred_np = prepared["x_pred"]
    W = prepared["W"]
    y = prepared["y"]
    noise = prepared["noise"]
    use_data_cov = prepared["use_data_cov"]
    ridge = prepared["ridge"]
    lam = prepared["lam"]
    max_train_points = prepared["max_train_points"]
    max_pred_points = prepared["max_pred_points"]

    device = torch.device(cfg.get("nn", {}).get("device", "cpu"))
    dtype = torch.float32

    xgrid_torch = torch.tensor(xgrid, device=device, dtype=dtype).unsqueeze(1)
    x_pred_torch = torch.tensor(x_pred_np, device=device, dtype=dtype).unsqueeze(1)
    W_torch = torch.tensor(W, device=device, dtype=dtype)
    y_torch = torch.tensor(y, device=device, dtype=dtype).reshape(-1)
    noise_torch = torch.tensor(noise, device=device, dtype=dtype)

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

    pred = ntk_gp_predict(
        model,
        xgrid_torch=xgrid_torch,
        W_train=W_torch,
        y_train=y_torch,
        x_pred_torch=x_pred_torch,
        noise_train=noise_torch,
        ridge=ridge,
        max_train_points=max_train_points,
        max_pred_points=max_pred_points,
    )

    f_mean = pred["f_mean"].detach().cpu().numpy().astype(np.float64)
    f_var = pred["f_var"].detach().cpu().numpy().astype(np.float64)
    if "f_cov" in pred and pred["f_cov"] is not None:
        f_cov = _symmetrize(pred["f_cov"].detach().cpu().numpy().astype(np.float64))
    else:
        f_cov = np.diag(f_var)

    out = _summary_from_gp_arrays(
        x_pred=x_pred_np,
        f_mean=f_mean,
        f_cov=f_cov,
        use_data_cov=use_data_cov,
        lam=lam,
        ridge=ridge,
        mode="empirical_ref",
    )
    if pred["train_sub_idx"] is not None:
        out["ntk_train_sub_idx"] = pred["train_sub_idx"].detach().cpu().numpy()
    if pred["pred_sub_idx"] is not None:
        out["ntk_pred_sub_idx"] = pred["pred_sub_idx"].detach().cpu().numpy()
    return out


def _build_neural_tangents_kernel_fn(cfg: Dict[str, Any]):
    from jax import numpy as jnp
    from neural_tangents import stax

    nncfg = cfg.get("nn", {})
    hidden = list(nncfg.get("hidden", [64, 64]))
    activation = str(nncfg.get("activation", "tanh")).lower()

    act_map = {
        "relu": stax.Relu(),
        "gelu": stax.Gelu(),
        "tanh": stax.ElementwiseNumerical(jnp.tanh, deg=25),
    }
    if activation not in act_map:
        raise ValueError(f"Unsupported activation for analytic infinite-width NTK: {activation!r}")

    layers = []
    for _ in hidden:
        layers += [stax.Dense(1), act_map[activation]]
    layers += [stax.Dense(1)]
    _, _, kernel_fn = stax.serial(*layers)
    return kernel_fn


def run_ntk_analytic_inf(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Analytic infinite-width NTK GP / kernel regression using Neural Tangents."""
    if not bool(cfg.get("ntk", {}).get("gp_eval", False)):
        return {}

    prepared = _prepare_standalone_ntk_inputs(ds, cfg)
    xgrid = prepared["xgrid"]
    x_pred = prepared["x_pred"]
    W = prepared["W"]
    y = prepared["y"]
    noise = prepared["noise"]
    use_data_cov = prepared["use_data_cov"]
    ridge = prepared["ridge"]
    lam = prepared["lam"]
    max_train_points = prepared["max_train_points"]
    max_pred_points = prepared["max_pred_points"]
    seed = prepared["seed"]

    idx_tr = _subsample_rows_np(W.shape[0], max_train_points, seed=seed)
    if idx_tr is not None:
        W_used = W[idx_tr]
        y_used = y[idx_tr]
        noise_used = noise[np.ix_(idx_tr, idx_tr)]
    else:
        W_used = W
        y_used = y
        noise_used = noise

    idxp = _subsample_pred_linspace_np(x_pred.shape[0], max_pred_points)
    x_pred_used = x_pred[idxp] if idxp is not None else x_pred

    kernel_fn = _build_neural_tangents_kernel_fn(cfg)

    xgrid_2d = xgrid[:, None]
    xpred_2d = x_pred_used[:, None]

    K_grid_grid = np.asarray(kernel_fn(xgrid_2d, xgrid_2d, get="ntk"), dtype=np.float64)
    K_grid_pred = np.asarray(kernel_fn(xgrid_2d, xpred_2d, get="ntk"), dtype=np.float64)
    K_pred_pred = np.asarray(kernel_fn(xpred_2d, xpred_2d, get="ntk"), dtype=np.float64)

    # Enforce NN-style scaling:
    #   f(x) = pre(x) * g(x),  pre(x) = x^alpha (1-x)^beta
    # so the physical kernel is K_f(x,x') = pre(x) K_g(x,x') pre(x')
    apply_scaling = bool(cfg.get("ntk", {}).get("apply_scaling", False))
    if apply_scaling:
        pre_grid = _ntk_prefactor_scaling(xgrid, cfg)
        pre_pred = _ntk_prefactor_scaling(x_pred_used, cfg)

        K_grid_grid = (pre_grid[:, None] * K_grid_grid) * pre_grid[None, :]
        K_grid_pred = (pre_grid[:, None] * K_grid_pred) * pre_pred[None, :]
        K_pred_pred = (pre_pred[:, None] * K_pred_pred) * pre_pred[None, :]

    K_grid_grid = _symmetrize(K_grid_grid)
    K_pred_pred = _symmetrize(K_pred_pred)

    K_yy = W_used @ K_grid_grid @ W_used.T
    K_yf = W_used @ K_grid_pred
    K_ff = K_pred_pred

    f_mean, _, f_cov = _gp_posterior_np(K_yy, K_yf, K_ff, y_used, noise_used, ridge)
    out = _summary_from_gp_arrays(
        x_pred=x_pred_used,
        f_mean=f_mean,
        f_cov=f_cov,
        use_data_cov=use_data_cov,
        lam=lam,
        ridge=ridge,
        mode="analytic_inf",
    )
    out["ntk_scaling_prefactor"] = "x^alpha(1-x)^beta"
    out["ntk_init_alpha"] = float(cfg.get("nn", {}).get("init_alpha", 1.0))
    out["ntk_init_beta"] = float(cfg.get("nn", {}).get("init_beta", 1.0))

    if idx_tr is not None:
        out["ntk_train_sub_idx"] = idx_tr
    if idxp is not None:
        out["ntk_pred_sub_idx"] = idxp
    return out


def run_ntk(ds: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone NTK model mode.

    Modes:
      - analytic_inf: true analytic infinite-width NTK route via Neural Tangents
      - empirical_ref: preserved finite-width empirical init implementation
      - auto: try analytic_inf first, optionally fall back to empirical_ref
    """
    ntk_cfg = cfg.get("ntk", {})
    mode = str(ntk_cfg.get("mode", "analytic_inf")).lower()
    fallback = bool(ntk_cfg.get("fallback_to_empirical_ref", True))

    if mode == "analytic_inf":
        try:
            return run_ntk_analytic_inf(ds, cfg)
        except ImportError:
            if fallback:
                return run_ntk_empirical_ref(ds, cfg)
            raise
    if mode == "empirical_ref":
        return run_ntk_empirical_ref(ds, cfg)
    if mode == "auto":
        try:
            return run_ntk_analytic_inf(ds, cfg)
        except ImportError:
            return run_ntk_empirical_ref(ds, cfg)

    raise ValueError(f"Unknown ntk.mode={mode!r}. Use 'analytic_inf', 'empirical_ref', or 'auto'.")
