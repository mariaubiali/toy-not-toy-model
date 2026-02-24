from __future__ import annotations

import os
import numpy as np
import arviz as az

from dataloader import load_dataset
from models.nuts import sample_hyperparams_nuts
from models.posterior import sample_replicas, posterior_fstar
from plotting import plot_fig1, plot_fig2, gp_uq_bands
# If you already have a fig2 plotting util, import it here:
# from utils.plotting import plot_fig2


def run_from_config(cfg: dict):
    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    # ----------------------------
    # Load + NUTS hyperparams
    # ----------------------------
    ds = load_dataset(cfg["data"])
    idata = sample_hyperparams_nuts(ds, cfg)

    # Save summary
    summ = az.summary(idata, var_names=["alpha", "l0", "sigma"])
    summ.to_csv(os.path.join(out, "summary.csv"))

    # ----------------------------
    # Prepare theta samples
    # ----------------------------
    alpha_s = idata.posterior["alpha"].values.ravel()
    l0_s = idata.posterior["l0"].values.ravel()
    sigma2_s = idata.posterior["sigma2"].values.ravel()
    theta_samples = np.stack([alpha_s, l0_s, sigma2_s], axis=1)

    # ----------------------------
    # Fig1 (trace / pairs etc.)
    # ----------------------------
    if cfg.get("eval", {}).get("make_fig1", True):
        fig1 = plot_fig1(idata)
        fig1.savefig(os.path.join(out, "fig1.pdf"))

    # ----------------------------
    # Posterior sampling of f*
    # ----------------------------
    if cfg.get("eval", {}).get("make_fig2", False):
        x_star_n = int(cfg.get("eval", {}).get("x_star_n", 400))
        # x_star = np.linspace(0.0, 1.0, x_star_n)
        

        X_train = ds["xgrid"].reshape(-1, 1)
        # X_star = x_star.reshape(-1, 1)
        X_star = X_train
        x_star = X_train.reshape(-1)
        print(x_star.shape)

        kcfg = cfg.get("kernel", {})
        delta = float(kcfg.get("delta", 1e-5))
        x_floor = float(kcfg.get("x_floor", 1e-12))
        jitter = float(kcfg.get("jitter", 1e-10))
        jitter_star = float(kcfg.get("jitter_star", 1e-10))
        nu = float(kcfg.get("nu", 1.5))    # only for matern kernel

        # Optional speed control: subsample theta draws
        max_s = cfg.get("eval", {}).get("fstar_max_samples", None)
        max_s = int(max_s) if max_s is not None else None

        # Optional: allow switching kernel for f* sampling
        # ("gibbs" default; can set eval.fstar_kernel: rbf)
        fstar_kernel = str(cfg.get("kernel", {}).get("name", "gibbs")).lower()

        replicas, means, vars_f = sample_replicas(
            theta_samples,
            X_train=X_train,
            X_star=X_star,
            FK=ds["W"],
            CY=ds["C"],
            y=ds["y"],
            kernel=fstar_kernel,
            seed=int(cfg.get("seed", 0)),
            delta=delta,
            x_floor=x_floor,
            jitter_cyt=jitter,
            jitter_star=jitter_star,
            nu = nu,
            max_samples=max_s,
        )

        # Save arrays for later plotting/postprocessing
        np.save(os.path.join(out, "x_star.npy"), x_star)
        np.save(os.path.join(out, "fstar_replicas.npy"), replicas)
        np.save(os.path.join(out, "fstar_means.npy"), means)

        # If you have a plot_fig2 util, you can call it here
        mean_curve = replicas.mean(axis=0)
        lo68, hi68 = np.percentile(replicas, [16, 84], axis=0)
        lo95, hi95 = np.percentile(replicas, [2.5, 97.5], axis=0)

        xt3_true = np.asarray(ds["meta"]["xt3_true"], float)
        xt3_true_star = np.interp(x_star, X_train[:, 0], xt3_true)

        plot_fig2(x_star, mean_curve, lo68, hi68, lo95, hi95, xt3_true_star=xt3_true_star, outpath=os.path.join(out, "fig2.pdf"))

        uq = gp_uq_bands(means, vars_f, sigma2_obs_star=0.0, draws_per_theta=1, seed=0)

        mean_curve = uq["mean_curve"]
        # lo68, hi68 = uq["pi_f_68"]      # (2, M) -> unpack
        # lo95, hi95 = uq["pi_f_95"]
        lo68, hi68, lo95, hi95 = uq["bands_f_mm"]
        plot_fig2(x_star, mean_curve, lo68, hi68, lo95, hi95, xt3_true_star=xt3_true_star, outpath=os.path.join(out, "uncbands.pdf"))



        # Save summary arrays for later bias/coverage checks
        save_replicas = bool(cfg.get("eval", {}).get("save_fstar_replicas", True))

        npz_path = os.path.join(out, "fstar_posterior_summary.npz")
        np.savez(
            npz_path,
            x_star=x_star,
            mean_curve=mean_curve,
            lo68=lo68,
            hi68=hi68,
            lo95=lo95,
            hi95=hi95,
            xt3_true_star=xt3_true_star if xt3_true_star is not None else np.array([]),
            # optional heavy arrays:
            replicas=replicas if save_replicas else np.array([]),
            means=means if save_replicas else np.array([]),
            vars_f=vars_f if save_replicas else np.array([]),
        )
        print(f"Saved f* posterior summary to {npz_path}")

    print(f"Done. Outputs in {out}")
    return idata
