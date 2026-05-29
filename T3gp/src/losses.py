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


def _safe_cholesky(
    A: torch.Tensor,
    jitter_abs: float = 1e-12,
    jitter_rel: float = 1e-8,
    max_tries: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Robust Cholesky with adaptive diagonal jitter.
    Returns (L, used_jitter).
    """
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)

    diag_mean = torch.mean(torch.diag(A)).detach()
    diag_mean = torch.nan_to_num(diag_mean, nan=1.0, posinf=1.0, neginf=1.0)
    base = torch.tensor(
        float(jitter_abs), device=A.device, dtype=A.dtype
    ) + torch.tensor(float(jitter_rel), device=A.device, dtype=A.dtype) * torch.abs(
        diag_mean
    )
    base = torch.clamp(base, min=torch.tensor(float(jitter_abs), device=A.device, dtype=A.dtype))

    for k in range(max_tries):
        jitter = base * (10.0**k)
        L, info = torch.linalg.cholesky_ex(A + jitter * eye)
        if int(info.max().item()) == 0:
            return L, jitter

    # Last attempt with a large jitter; surface the original linalg error if still failing.
    jitter = base * (10.0**max_tries)
    L = torch.linalg.cholesky(A + jitter * eye)
    return L, jitter

def pointwise_loss_members(
    y_pred_members: torch.Tensor,
    y: torch.Tensor,
    *,
    loss_name: str,
    C: Optional[torch.Tensor] = None,
    L: Optional[torch.Tensor] = None,
    jitter: float = 1e-10,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Return per-member, per-data-point loss profiles for ensemble methods.

    Parameters
    ----------
    y_pred_members:
        Tensor with shape (K, Ndat), where K is the number of ensemble members.
    y:
        Tensor with shape (Ndat,).
    loss_name:
        One of ``"mse"``, ``"weighted_mse"``, or ``"chi2"``.
    C:
        Experimental covariance matrix with shape (Ndat, Ndat). Required for
        ``weighted_mse`` and ``chi2`` unless an appropriate Cholesky factor is
        supplied for ``chi2``.
    L:
        Optional Cholesky factor of C, with C = L L^T. Used for ``chi2``.

    Returns
    -------
    per_point_loss:
        Tensor with shape (K, Ndat). For ``chi2`` this is the squared whitened
        residual, so summing along axis 1 gives each member's chi-square.
    """
    loss_name = str(loss_name).lower()

    if y_pred_members.ndim != 2:
        raise ValueError(
            "y_pred_members must have shape (K, Ndat), "
            f"got {tuple(y_pred_members.shape)}"
        )

    y = y.to(device=y_pred_members.device, dtype=y_pred_members.dtype).reshape(-1)
    if y_pred_members.shape[1] != y.numel():
        raise ValueError(
            "y_pred_members and y have inconsistent data dimensions: "
            f"{y_pred_members.shape[1]} vs {y.numel()}"
        )

    residual = y_pred_members - y[None, :]

    if loss_name == "mse":
        return residual.square()

    if loss_name == "weighted_mse":
        if C is None:
            raise ValueError("weighted_mse pointwise loss requires C.")
        C = C.to(device=y_pred_members.device, dtype=y_pred_members.dtype)
        var = torch.diag(C).clamp_min(float(eps))
        return residual.square() / var[None, :]

    if loss_name == "chi2":
        if L is None:
            if C is None:
                raise ValueError("chi2 pointwise loss requires C or precomputed L.")
            C = C.to(device=y_pred_members.device, dtype=y_pred_members.dtype)
            L = _cholesky_C(C, jitter)
        else:
            L = L.to(device=y_pred_members.device, dtype=y_pred_members.dtype)

        z = torch.linalg.solve_triangular(
            L,
            residual.T,
            upper=False,
        ).T
        return z.square()

    raise NotImplementedError(
        "Pointwise member losses currently support loss.name in "
        "{'mse', 'weighted_mse', 'chi2'}. "
        f"Got loss.name={loss_name!r}."
    )

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

            # if lambda_sr > 0.0:
            #     if "f_mu" not in extra:
            #         raise ValueError("chi2+sumrule requires extra['f_mu'].")
            #     f_raw = extra["f_mu"].reshape(-1)
            #     I_mid = torch.trapz(f_raw / xg, xg)
            #     loss = loss + lambda_sr * (I_mid - ref) ** 2

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
        full_cov = bool(lcfg.get("full_cov", False))
        full_cov_jitter_abs = float(lcfg.get("full_cov_jitter_abs", eps))
        full_cov_jitter_rel = float(lcfg.get("full_cov_jitter_rel", 1e-8))

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

            r = y_mu - y_true
            if full_cov:
                # Full y-space covariance from diagonal Sigma_f:
                # Cov_y = W Sigma_f W^T, with Sigma_f = diag(var_f).
                Cov_y = (W * var_f.unsqueeze(0)) @ W.T
                L, _ = _safe_cholesky(
                    Cov_y,
                    jitter_abs=full_cov_jitter_abs,
                    jitter_rel=full_cov_jitter_rel,
                )
                alpha = _apply_Cinv(L, r)
                quad = r @ alpha
                logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
                loss = 0.5 * (quad + logdet) / y_true.numel()
            else:
                # diagonal approximation in y-space
                var_y = (W * W) @ var_f
                var_y = var_y.clamp_min(eps)
                r2 = r * r
                nll = 0.5 * (r2 / var_y + torch.log(var_y))
                loss = nll.mean()

            # sum rule on f_mu
            # if lambda_sr > 0.0:
            #     I_mid = torch.trapz(f_mu / xg, xg)
            #     loss = loss + lambda_sr * (I_mid - ref) ** 2

            return loss

        return loss_fn, extra_params

    # --- T3_Beta-style chi^2 + sum-rule penalty  ---
    if name == "chi2":
        # load params from training module
        y = ctx.y.to(device=device, dtype=dtype).reshape(-1)  # cache once
        xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)  # cache once

        ref = (
            torch.tensor(float(ctx.t3_ref_int), device=device, dtype=dtype)
            if ctx.t3_ref_int is not None
            else None
        )

        if ctx.L is not None:
            L = ctx.L.to(device=device, dtype=dtype)
        else:
            L = _cholesky_C(ctx.C.to(device=device, dtype=dtype), ctx.jitter)

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            y_pred = y_pred.reshape(-1)
            resid = y_pred - y
            loss_chi2 = resid @ _apply_Cinv(L, resid)

            # if lambda_sr > 0.0:
            #     if "f_mu" not in extra:
            #         raise ValueError("chi2+sumrule requires extra['f_mu'].")
            #     f_raw = extra["f_mu"].reshape(-1)
            #     I_mid = torch.trapz(f_raw / xg, xg)
            #     loss_chi2 = loss_chi2 + lambda_sr * (I_mid - ref) ** 2

            return loss_chi2

        return loss_fn, extra_params

    if name == "chi2_het":
        eps = float(lcfg.get("eps", 1e-12))
        logvar_clip = lcfg.get("logvar_clip", (-20.0, 10.0))
        full_cov_jitter_abs = float(lcfg.get("full_cov_jitter_abs", 1e-12))
        full_cov_jitter_rel = float(lcfg.get("full_cov_jitter_rel", 1e-8))
        normalize = bool(lcfg.get("normalize_by_ndata", True))

        W = ctx.W.to(device=device, dtype=dtype)  # (Ndat, Ngrid)
        C = ctx.C.to(device=device, dtype=dtype)  # (Ndat, Ndat)
        y = ctx.y.to(device=device, dtype=dtype).reshape(-1)  # (Ndat,)
        xg = ctx.xgrid.to(device=device, dtype=dtype).reshape(-1)

        ref = (
            torch.tensor(float(ctx.t3_ref_int), device=device, dtype=dtype)
            if ctx.t3_ref_int is not None
            else None
        )

        def loss_fn(y_pred: torch.Tensor, extra: dict) -> torch.Tensor:
            y_pred = y_pred.reshape(-1)
            resid = y_pred - y
            if "f_mu" not in extra:
                raise ValueError("chi2_het requires extra['f_mu'].")
            if "f_logvar" not in extra:
                raise ValueError(
                    "chi2_het requires extra['f_logvar'] (grid log-variance). "
                    "Set out_dim=2 or ensure the model returns log-variance."
                )

            f_mu = extra["f_mu"].to(device=device, dtype=dtype).reshape(-1)  # (Ngrid,)
            f_logvar = extra["f_logvar"].to(device=device, dtype=dtype).reshape(-1)
            lo, hi = float(logvar_clip[0]), float(logvar_clip[1])
            f_logvar = f_logvar.clamp(lo, hi)
            var_f = torch.exp(f_logvar).clamp_min(eps)  # (Ngrid,)


            # # Use a self-consistent predictive mean in data space
            y_mu = W @ f_mu  # (Ndat,)
            r = y_mu - y

            # Full propagated model covariance in data space:
            # Cov_model = W diag(var_f) W^T
            Cov_model = (W * var_f.unsqueeze(0)) @ W.T  # (Ndat, Ndat)

            # Total covariance = experimental + model
            Cov_total = C + Cov_model
            Cov_total = 0.5 * (Cov_total + Cov_total.T)

            L, _ = _safe_cholesky(
                Cov_total,
                jitter_abs=full_cov_jitter_abs,
                jitter_rel=full_cov_jitter_rel,
            )

            alpha = _apply_Cinv(L, r)
            quad = r @ alpha
            logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))

            loss = 0.5 * (quad + logdet)
            if normalize:
                loss = loss / y.numel()

            # if lambda_sr > 0.0:
            #     I_mid = torch.trapz(f_mu / xg, xg)
            #     loss = loss + lambda_sr * (I_mid - ref) ** 2

            return loss

        return loss_fn, extra_params

    raise ValueError(f"Unknown loss.name: {name}")
