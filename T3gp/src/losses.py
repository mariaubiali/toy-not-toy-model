from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LossContext:
    """
    Context passed to the loss functions.
    """
    W: torch.Tensor              # (Ndat, Ngrid)
    C: torch.Tensor              # (Ndat, Ndat)
    y: torch.Tensor              # (Ndat,)
    xgrid: Optional[torch.Tensor] = None   # (Ngrid,) needed for T3_Beta sum-rule
    t3_ref_int: float | None = None
    jitter: float = 1e-10


def _cholesky_C(C: torch.Tensor, jitter: float) -> torch.Tensor:
    C = 0.5 * (C + C.T)
    C = C + jitter * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    return torch.linalg.cholesky(C)


def _apply_Cinv(L: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # solve C^{-1} r using Cholesky L (C = L L^T)
    # r: (Ndat,) or (Ndat,1)
    if r.ndim == 1:
        r = r[:, None]
    v = torch.linalg.solve_triangular(L, r, upper=False)
    x = torch.linalg.solve_triangular(L.T, v, upper=True)
    return x[:, 0]


def make_loss(
    cfg: Dict[str, Any],
    ctx: LossContext,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Returns (loss_fn, extra_params)

    loss_fn signature:
        loss_fn(y_pred: Tensor(Ndat,), extra: dict) -> scalar Tensor

    extra_params is an nn.ParameterDict you can hand to the optimizer (may be empty).
    """

    lcfg = cfg.get("loss", {})
    name = str(lcfg.get("name", "weighted_mse")).lower()
    lambda_sr = float(lcfg.get("lambda_sr", 0.0))
    if ctx.xgrid is None:
        raise ValueError("t3_beta/chi2_sumrule requires ctx.xgrid (1D tensor of the x-grid).")
    xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)

    extra_params = nn.ParameterDict()

    # Precompute Cholesky for weighted losses
    L = None
    if name in {"weighted_mse"}:
        L = _cholesky_C(ctx.C.to(device=device, dtype=dtype), ctx.jitter)

    if name == "mse":
        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            r = y_pred - ctx.y.to(device=device, dtype=dtype)

            loss_sumrule = torch.tensor(0.0, device=device, dtype=dtype)
            if lambda_sr > 0.0:
                if "f_grid" not in extra:
                    raise ValueError("t3_beta/chi2_sumrule requires model output extra['f_grid'].")
                f_raw = extra["f_grid"].to(device=device, dtype=dtype).reshape(-1)

                # Exactly as in T3_Beta (no clamping)
                t3_unc = f_raw / xg
                I_mid = torch.trapz(t3_unc, xg)

                ref = torch.tensor(ctx.t3_ref_int, device=device, dtype=dtype)
                loss_sumrule = lambda_sr * (I_mid - ref) ** 2
            return torch.mean(r * r) + loss_sumrule
        return loss_fn, extra_params

    if name == "weighted_mse":
        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            r = y_pred - ctx.y.to(device=device, dtype=dtype)
            Cinvr = _apply_Cinv(L, r)
            return torch.mean(r * Cinvr)
        return loss_fn, extra_params
    
    if name == "mse_het":

        eps = float(lcfg.get("eps", 1e-12))
        logvar_clip = lcfg.get("logvar_clip", (-20.0, 10.0))

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            """
            y_pred[..., 0] = mean prediction μ
            y_pred[..., 1] = log variance log(σ²)
            """

            y_true = ctx.y.to(device=device, dtype=dtype)

            mu = y_pred[..., 0]
            logvar = y_pred[..., 1]

            # numerical safety
            lo, hi = float(logvar_clip[0]), float(logvar_clip[1])
            logvar = logvar.clamp(lo, hi)

            # σ² = exp(log σ²)
            var = torch.exp(logvar).clamp_min(eps)

            r2 = (mu - y_true) ** 2

            # Gaussian NLL (heteroscedastic MSE)
            loss = 0.5 * (r2 / var + logvar)

            return loss.mean()

        return loss_fn, extra_params

    # --- T3_Beta-style chi^2 + sum-rule penalty  ---
    if name in {"t3_beta", "chi2"}:

        # Use Cholesky solves for C^{-1}r (more efficient/stable than forming C^{-1}).
        C_t = ctx.C.to(device=device, dtype=dtype)
        L = _cholesky_C(C_t, ctx.jitter)

        if ctx.xgrid is None:
            raise ValueError("t3_beta/chi2_sumrule requires ctx.xgrid (1D tensor of the x-grid).")
        xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            y = ctx.y.to(device=device, dtype=dtype).reshape(-1)
            y_pred = y_pred.reshape(-1)

            resid = y_pred - y
            Cinv_r = _apply_Cinv(L, resid)
            loss_chi2 = resid @ Cinv_r

            loss_sumrule = torch.tensor(0.0, device=device, dtype=dtype)
            if lambda_sr > 0.0:
                if "f_grid" not in extra:
                    raise ValueError("t3_beta/chi2_sumrule requires model output extra['f_grid'].")
                f_raw = extra["f_grid"].to(device=device, dtype=dtype).reshape(-1)

                # Exactly as in T3_Beta (no clamping)
                t3_unc = f_raw / xg
                I_mid = torch.trapz(t3_unc, xg)

                ref = torch.tensor(ctx.t3_ref_int, device=device, dtype=dtype)
                loss_sumrule = lambda_sr * (I_mid - ref) ** 2

            return loss_chi2 + loss_sumrule

        return loss_fn, extra_params
    
    if name in ("chi_het"):
        # Sum-rule settings (same as T3_Beta)

        # Heteroscedastic settings
        eps = float(lcfg.get("eps", 1e-12))
        logvar_clip = lcfg.get("logvar_clip", (-20.0, 10.0))
        # choose either: "nll" (Gaussian NLL) or "chi2" (scaled SSE)
        form = str(lcfg.get("form", "nll")).lower()

        W = ctx.W.to(device=device, dtype=dtype)
        y = ctx.y.to(device=device, dtype=dtype)
        xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)

        # observational (data) variance baseline from diagonal of covariance
        diagC = torch.diag(ctx.C.to(device=device, dtype=dtype)).clamp_min(eps)

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            # y_pred expected to be W @ f_mean (built by trainer)
            if "logvar_f_grid" not in extra:
                raise ValueError(
                    "t3_beta_het requires model output logvar_f_grid "
                    "(set model out_dim>=2)."
                )
            f_mean = extra["f_grid"].to(device=device, dtype=dtype).reshape(-1)

            # predicted variance on f-grid
            logvar = extra["logvar_f_grid"].to(device=device, dtype=dtype)
            lo, hi = float(logvar_clip[0]), float(logvar_clip[1])
            logvar = logvar.clamp(lo, hi).reshape(-1)
            var_f = torch.exp(logvar).clamp_min(eps)  # (Ngrid,)

            # propagate to y-space (diagonal approx):
            # Var(W f) ~= sum_j W_ij^2 Var(f_j)
            var_y = diagC + (W * W) @ var_f           # (Ndat,)

            r = (y_pred - y).reshape(-1)

            if form == "chi2":
                data_term = torch.mean((r * r) / var_y)
            elif form == "mse":
                # plain MSE but still learning logvar won't help; included for completeness
                data_term = torch.mean(r * r)
            else:
                # Gaussian negative log-likelihood (heteroscedastic regression)
                data_term = 0.5 * torch.mean((r * r) / var_y + torch.log(var_y))

            # Sum rule penalty (identical to T3_Beta)
            sumrule = torch.tensor(0.0, device=device, dtype=dtype)
            if lambda_sr > 0.0:
                t3_unc = f_mean / xg
                I_mid = torch.trapz(t3_unc, xg)

                ref = torch.tensor(ctx.t3_ref_int, device=device, dtype=dtype)
                sumrule = lambda_sr * (I_mid - ref) ** 2

            return data_term + sumrule

        return loss_fn, extra_params

    raise ValueError(f"Unknown loss.name: {name}")
