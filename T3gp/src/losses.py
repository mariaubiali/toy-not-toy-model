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

    W: torch.Tensor  # (Ndat, Ngrid)
    C: torch.Tensor  # (Ndat, Ndat)
    y: torch.Tensor  # (Ndat,)
    xgrid: Optional[torch.Tensor] = None  # (Ngrid,) needed for T3_Beta sum-rule
    t3_ref_int: float | None = None
    jitter: float = 1e-10
    L: Optional[torch.Tensor] = None  # Precomputed Cholesky of C (Ndat, Ndat)


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
        raise ValueError("loss requires ctx.xgrid (1D tensor of the x-grid).")

    # cache xgrid once (used by several losses)
    xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)

    extra_params = nn.ParameterDict()

    # ---- MSE loss ----
    if name == "mse":
        y = ctx.y.to(device=device, dtype=dtype).reshape(-1)
        ref = (
            torch.tensor(float(ctx.t3_ref_int), device=device, dtype=dtype)
            if ctx.t3_ref_int is not None
            else None
        )

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            y_pred = y_pred.reshape(-1)
            r = y_pred - y
            loss = torch.mean(r * r)

            if lambda_sr > 0.0:
                if "f_mu" not in extra:
                    raise ValueError("chi2+sumrule requires extra['f_mu'].")
                f_raw = extra["f_mu"].reshape(-1)
                I_mid = torch.trapz(f_raw / xg, xg)
                loss = loss + lambda_sr * (I_mid - ref) ** 2

            return loss

        return loss_fn, extra_params

    # ---- Weighted MSE loss ----
    if name == "weighted_mse":
        y = ctx.y.to(device=device, dtype=dtype).reshape(-1)
        L = _cholesky_C(ctx.C.to(device=device, dtype=dtype), ctx.jitter)

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            y_pred = y_pred.reshape(-1)
            r = y_pred - ctx.y.to(device=device, dtype=dtype)
            Cinvr = _apply_Cinv(L, r)
            return torch.mean(r * Cinvr)

        return loss_fn, extra_params

    if name == "mse_het":
        eps = float(lcfg.get("eps", 1e-12))
        logvar_clip = lcfg.get("logvar_clip", (-20.0, 5.0))  # tighter hi often helps

        y_true = ctx.y.to(device=device, dtype=dtype).reshape(-1)
        W = ctx.W.to(device=device, dtype=dtype)  # (Ntr, Ngrid)
        xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)
        ref = (
            torch.tensor(float(ctx.t3_ref_int), device=device, dtype=dtype)
            if ctx.t3_ref_int is not None
            else None
        )

        def loss_fn(y_mu: torch.Tensor, extra: dict) -> torch.Tensor:
            if "f_logvar" not in extra:
                raise ValueError(
                    "mse_het requires extra['f_logvar'] (grid log-variance). "
                    "Your model output seems to be out_dim=1 or not providing logvar."
                )

            # y_mu is the predicted mean in data space (Ntr,)
            y_mu = y_mu.reshape(-1)

            f_mu = extra["f_mu"].to(device=device, dtype=dtype).reshape(-1)  # (Ngrid,)
            f_logvar = (
                extra["f_logvar"].to(device=device, dtype=dtype).reshape(-1)
            )  # (Ngrid,)

            lo, hi = float(logvar_clip[0]), float(logvar_clip[1])
            f_logvar = f_logvar.clamp(lo, hi)
            var_f = torch.exp(f_logvar).clamp_min(eps)  # (Ngrid,)

            # propagate grid variance to data variance
            var_y = (W * W) @ var_f
            var_y = var_y.clamp_min(eps)

            r2 = (y_mu - y_true) ** 2
            nll = 0.5 * (r2 / var_y + torch.log(var_y))
            loss = nll.mean()

            # sum rule on f_mu
            if lambda_sr > 0.0:
                I_mid = torch.trapz(f_mu / xg, xg)
                loss = loss + lambda_sr * (I_mid - ref) ** 2

            return loss

        return loss_fn, extra_params

    # --- T3_Beta-style chi^2 + sum-rule penalty  ---
    if name == "chi2":
        # load params from training module
        y = ctx.y.to(device=device, dtype=dtype).reshape(-1)  # cache once
        xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)  # cache once

        ref = torch.tensor(ctx.t3_ref_int, device=device, dtype=dtype)  # cache once

        if ctx.L is not None:
            L = ctx.L.to(device=device, dtype=dtype)
        else:
            L = _cholesky_C(ctx.C.to(device=device, dtype=dtype), ctx.jitter)

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            y_pred = y_pred.reshape(-1)
            resid = y_pred - y
            loss_chi2 = resid @ _apply_Cinv(L, resid)

            if lambda_sr > 0.0:
                if "f_mu" not in extra:
                    raise ValueError("chi2+sumrule requires extra['f_mu'].")
                f_raw = extra["f_mu"].reshape(-1)
                I_mid = torch.trapz(f_raw / xg, xg)

            return loss_chi2 + lambda_sr * (I_mid - ref) ** 2

        return loss_fn, extra_params

    if name == "chi_het":
        eps = float(lcfg.get("eps", 1e-12))
        logvar_clip = lcfg.get("logvar_clip", (-20.0, 10.0))

        W = ctx.W.to(device=device, dtype=dtype)  # (Ntr, Ngrid)
        y = ctx.y.to(device=device, dtype=dtype).reshape(-1)  # (Ntr,)
        xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)

        # baseline observational variance from diagonal of C (train block)
        diagC = torch.diag(ctx.C.to(device=device, dtype=dtype)).clamp_min(
            eps
        )  # (Ntr,)

        ref = (
            torch.tensor(float(ctx.t3_ref_int), device=device, dtype=dtype)
            if ctx.t3_ref_int is not None
            else None
        )

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            # y_pred expected to be W @ f_mu (built by trainer)
            y_pred = y_pred.reshape(-1)

            if "f_mu" not in extra:
                raise ValueError("chi_het requires extra['f_mu'] (grid mean).")
            if "f_logvar" not in extra:
                raise ValueError(
                    "chi_het requires extra['f_logvar'] (grid log-variance). Set out_dim=2."
                )

            f_mu = extra["f_mu"].to(device=device, dtype=dtype).reshape(-1)  # (Ngrid,)
            f_logvar = (
                extra["f_logvar"].to(device=device, dtype=dtype).reshape(-1)
            )  # (Ngrid,)

            lo, hi = float(logvar_clip[0]), float(logvar_clip[1])
            f_logvar = f_logvar.clamp(lo, hi)
            var_f = torch.exp(f_logvar).clamp_min(eps)  # (Ngrid,)

            # propagate to y-space (diag approx) and add experimental diag(C)
            var_y = diagC + (W * W) @ var_f  # (Ntr,)
            var_y = var_y.clamp_min(eps)

            r = y_pred - y

            # Gaussian NLL using var_y
            data_term = 0.5 * torch.mean((r * r) / var_y + torch.log(var_y))

            # sum rule penalty on f_mu
            sumrule = torch.tensor(0.0, device=device, dtype=dtype)
            if lambda_sr > 0.0:
                I_mid = torch.trapz(f_mu / xg, xg)
                sumrule = lambda_sr * (I_mid - ref) ** 2

            return data_term + sumrule

        return loss_fn, extra_params

    raise ValueError(f"Unknown loss.name: {name}")
