from __future__ import annotations

from typing import Any, Dict
import os

import pymc as pm
import pytensor.tensor as pt
import arviz as az

from models.gibbs import Kxx_gibbs_pytensor, build_log_marginal_likelihood_pt
from models.rbf import Kxx_rbf_pytensor
from models.matern import Kxx_matern_pytensor


def _prior(name: str, spec: Dict[str, Any]):
    if spec["dist"].lower() == "uniform":
        return pm.Uniform(name, lower=float(spec["low"]), upper=float(spec["high"]))
    raise ValueError(f"Unsupported prior spec: {spec}")


def sample_hyperparams_nuts(dataset: Dict[str, Any], cfg: Dict[str, Any]):
    xgrid = dataset["xgrid"]
    W = dataset["W"]
    C = dataset["C"]
    y = dataset["y"]

    xgrid_t = pt.constant(xgrid)
    W_t = pt.constant(W)
    C_t = pt.constant(C)
    y_t = pt.constant(y)

    kcfg = cfg["kernel"]
    kname = str(kcfg.get("name", "gibbs")).lower()

    delta = float(kcfg.get("delta", 1e-5))
    x_floor = float(kcfg.get("x_floor", 1e-12))
    jitter = float(kcfg.get("jitter", 1e-10))
    nu = float(kcfg.get("nu", 1.5))    # only for matern kernel

    def Kxx_fn(xgrid_t_, alpha, l0, sigma2):
        if kname == "gibbs":
            return Kxx_gibbs_pytensor(
                xgrid_t_, alpha, l0, sigma2, delta=delta, x_floor=x_floor
            )
        elif kname == "rbf":
            return Kxx_rbf_pytensor(xgrid_t_, alpha, l0, sigma2, x_floor=x_floor)
        elif kname == "matern":
            return Kxx_matern_pytensor(
                xgrid_t_, alpha, l0, sigma2, nu=nu, x_floor=x_floor
            )
        else:
            raise ValueError(f"Unknown kernel.name={kname!r}")

    lml = build_log_marginal_likelihood_pt(xgrid_t, W_t, C_t, y_t, Kxx_fn, jitter=jitter)

    pri = cfg["priors"]
    nuts = cfg["nuts"]
    seed = int(cfg.get("seed", 0))
    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    with pm.Model() as model:
        alpha = _prior("alpha", pri["alpha"])
        l0 = _prior("l0", pri["l0"])
        sigma = _prior("sigma", pri["sigma"])
        sigma2 = pm.Deterministic("sigma2", sigma**2)

        pm.Potential("logpost", lml(alpha, l0, sigma2))

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