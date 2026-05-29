from __future__ import annotations

from typing import Any, Dict
import os

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from models.gibbs import Kxx_gibbs_pytensor, build_log_marginal_likelihood_pt
from models.rbf import Kxx_rbf_pytensor
from models.matern import Kxx_matern_pytensor
from transforms import log_x_gp


def _prior(name: str, spec: Dict[str, Any]):
    if spec["dist"].lower() == "uniform":
        return pm.Uniform(name, lower=float(spec["low"]), upper=float(spec["high"]))
    raise ValueError(f"Unsupported prior spec: {spec}")


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("Need at least 2 points for trapezoid weights.")
    w = np.zeros_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def _pre_pt(
    x_phys_t: pt.TensorVariable,
    alpha: pt.TensorVariable,
    beta: pt.TensorVariable | float,
    x_clip: float = 1e-12,
    x_floor: float = 1e-12,
) -> pt.TensorVariable:
    """
    Physical prefactor

        pre(x) = x^alpha (1 - x)^beta

    This must always be evaluated on physical Bjorken-x values, not on
    transformed kernel coordinates such as log(x).
    """
    if beta is None:
        raise ValueError("_pre_pt got beta=None but prefactor mode requires beta.")

    if not isinstance(beta, pt.TensorVariable):
        beta = pt.constant(float(beta))

    x = pt.clip(x_phys_t, x_clip, 1.0 - x_clip)
    x_s = pt.maximum(x, x_floor)

    return pt.power(x_s, alpha) * pt.power(1.0 - x, beta)


def sample_hyperparams_nuts(dataset: Dict[str, Any], cfg: Dict[str, Any]):
    # Physical Bjorken-x grid. This is the grid used for the prefactor
    # pre(x) = x^alpha (1-x)^beta.
    xgrid_phys = np.asarray(dataset["xgrid"], dtype=np.float64).reshape(-1)

    # Kernel-coordinate grid. This may be physical x or transformed x, e.g. log(x),
    # depending on cfg["transforms"].
    transforms = cfg.get("transforms", {})
    xgrid_kernel = log_x_gp(xgrid_phys, transforms)

    W = np.asarray(dataset["W"], dtype=np.float64)
    C = np.asarray(dataset["C"], dtype=np.float64)
    y = np.asarray(dataset["y"], dtype=np.float64).reshape(-1)

    xgrid_t = pt.constant(xgrid_kernel)
    xgrid_phys_t = pt.constant(xgrid_phys)

    W_t = pt.constant(W)
    C_t = pt.constant(C)
    y_t = pt.constant(y)
    y_t = pt.constant(y)

    kcfg = cfg["kernel"]
    kname = str(kcfg.get("name", "gibbs")).lower()

    print(f"Selected kernel for NUTS hyperparam inference: {kname}")

    delta = float(kcfg.get("delta", 1e-5))
    x_floor = float(kcfg.get("x_floor", 1e-12))
    jitter = float(kcfg.get("jitter", 1e-10))
    nu = float(kcfg.get("nu", 1.5))  # only for matern kernel
    lambda_sr = float(kcfg.get("lambda_sr", 0.0))
    pcfg = cfg.get("gp_prefactor", {})
    pref_mode = str(pcfg.get("mode", "prefactor")).lower()

    if pref_mode not in ("legacy", "prefactor", "none"):
        raise ValueError(
            f"Unknown gp_prefactor.mode={pref_mode!r} (use legacy|prefactor|none)"
        )

    # ----------------------------
    # Optional sumrule pseudo-observation (matches NN penalty)
    # ----------------------------
    # if lambda_sr > 0.0:
    #     meta = dataset.get("meta", {})
    #     xt3_true = np.asarray(meta.get("xt3_true", []), float).ravel()
    #     if xt3_true.size != xgrid.shape[0]:
    #         raise ValueError(
    #             "Need meta['xt3_true'] defined on full xgrid for sumrule ref integral."
    #         )

    #     # ref = ∫ (xt3_true/x) dx on full grid (same as NN)
    #     ref = float(np.trapz(xt3_true / xgrid, xgrid))

    #     # I(f) ≈ sum_i w_i * (f_i / x_i) = a^T f
    #     w = _trapz_weights(xgrid)
    #     a = (w / xgrid).astype(np.float64)  # (Ngrid,)
    #     # tau^2 chosen so 0.5*(I-ref)^2/tau^2 == lambda_sr*(I-ref)^2
    #     tau2 = 1.0 / (2.0 * lambda_sr)

    #     a_t = pt.constant(a)
    #     ref_t = pt.constant(np.array([ref], dtype=np.float64))
    #     tau2_t = pt.constant(float(tau2))

    #     # augment W: (Ndat+1, Ngrid)
    #     W_t = pt.concatenate([W_t, a_t[None, :]], axis=0)

    #     # augment y: (Ndat+1,)
    #     y_t = pt.concatenate([y_t, ref_t], axis=0)

    #     # augment C: blockdiag(C, tau2)
    #     n = C_t.shape[0]
    #     C_aug = pt.zeros((n + 1, n + 1), dtype=C_t.dtype)
    #     C_aug = pt.set_subtensor(C_aug[:n, :n], C_t)
    #     C_aug = pt.set_subtensor(C_aug[n, n], tau2_t)
    #     C_t = C_aug

    pri = cfg["priors"]
    nuts = cfg["nuts"]
    seed = int(cfg.get("seed", 0))
    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    with pm.Model() as model:
        # Always sample kernel hypers that always exist
        l0 = _prior("l0", pri["l0"])
        sigma = _prior("sigma", pri["sigma"])
        sigma2 = pm.Deterministic("sigma2", sigma**2)

        # Only sample alpha when scaling is used
        alpha_rv = None
        if pref_mode in ("legacy", "prefactor"):
            alpha_rv = _prior("alpha", pri["alpha"])

        # Only sample beta in prefactor mode
        beta_rv = None
        if pref_mode == "prefactor":
            if "beta" not in pri:
                raise KeyError(
                    "gp_prefactor.mode=prefactor requires priors.beta in runcard."
                )
            beta_rv = _prior("beta", pri["beta"])

        def _K0(xgrid_t_, l0_, sigma2_):
            """
            Base kernel with no external prefactor.

            The input xgrid_t_ is the kernel coordinate, e.g. log(x) if enabled.
            """
            alpha0 = pt.constant(0.0)

            if kname == "gibbs":
                return Kxx_gibbs_pytensor(
                    xgrid_t_,
                    alpha0,
                    l0_,
                    sigma2_,
                    delta=delta,
                    x_floor=x_floor,
                )

            elif kname == "rbf":
                return Kxx_rbf_pytensor(
                    xgrid_t_,
                    alpha0,
                    l0_,
                    sigma2_,
                    amp="none",
                    x_floor=x_floor,
                )

            elif kname == "matern":
                return Kxx_matern_pytensor(
                    xgrid_t_,
                    alpha0,
                    l0_,
                    sigma2_,
                    nu=nu,
                    x_floor=x_floor,
                )

            else:
                raise ValueError(f"Unknown kernel.name={kname!r}")

        def Kxx_fn(xgrid_t_, alpha_, l0_, sigma2_):
            if pref_mode == "none":
                return _K0(xgrid_t_, l0_, sigma2_)

            if pref_mode == "legacy":
                raise ValueError(
                    "gp_prefactor.mode='legacy' is disabled for this setup. "
                    "Use gp_prefactor.mode='prefactor' for "
                    "K(x,y)=pre(x)pre(y)K0(x,y), with pre(x)=x^alpha(1-x)^beta."
                )

            if pref_mode == "prefactor":
                # Base kernel is evaluated in kernel-coordinate space, e.g. log(x).
                K0 = _K0(xgrid_t_, l0_, sigma2_)

                # The prefactor is evaluated in physical x-space.
                pre = _pre_pt(
                    xgrid_phys_t,
                    alpha_,
                    beta_rv,
                    x_clip=1e-12,
                    x_floor=x_floor,
                )

                return (pre[:, None] * K0) * pre[None, :]

            raise ValueError(f"Unknown gp_prefactor.mode={pref_mode!r}")

        lml = build_log_marginal_likelihood_pt(
            xgrid_t, W_t, C_t, y_t, Kxx_fn, jitter=jitter
        )

        # IMPORTANT: call lml with the correct alpha value depending on mode
        if pref_mode == "none":
            pm.Potential("logpost", lml(pt.constant(0.0), l0, sigma2))
        else:
            pm.Potential("logpost", lml(alpha_rv, l0, sigma2))

        idata = pm.sample(
            draws=int(nuts.get("draws", 1000)),
            tune=int(nuts.get("tune", 1000)),
            chains=int(nuts.get("chains", 2)),
            cores=int(nuts.get("cores", 1)),
            target_accept=float(nuts.get("target_accept", 0.9)),
            init=str(nuts.get("init", "jitter+adapt_diag")),
            random_seed=seed,
        )

    if cfg.get("eval", {}).get("save_trace_nc", True):
        az.to_netcdf(idata, os.path.join(out, "posterior_trace.nc"))

    return idata
