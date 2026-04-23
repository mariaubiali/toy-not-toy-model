from __future__ import annotations

import os
import numpy as np
import arviz as az
from sklearn.model_selection import train_test_split

from dataloader import load_dataset
from models.nuts import sample_hyperparams_nuts
from models.posterior import sample_replicas
from plotting import plot_fig1, plot_fig2, gp_uq_bands
from transforms import log_x_gp


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("Need at least 2 points for trapezoid weights.")
    w = np.zeros_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w

def ensure_symmetric(C):
    C = np.asarray(C, dtype=np.float64)
    return 0.5 * (C + C.T)

def run_from_config(cfg: dict):
    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    # ----------------------------
    # Load dataset
    # ----------------------------
    ds = load_dataset(cfg["data"])

    W_full = np.asarray(ds["W"], dtype=np.float64)                      # (Ndat, Nfk)
    C_full = np.asarray(ds["C"], dtype=np.float64)                      # (Ndat, Ndat)
    diagC = np.diag(C_full).astype(np.float64)
    y_pseudo = np.asarray(ds['y'], dtype=np.float64).ravel()

    # ----------------------------
    # Train/val split over DATA points (rows of W) – match NN
    # ----------------------------
    n_data = W_full.shape[0]
    replica = int(cfg.get("replica", 0))
    idx_all = np.arange(n_data)
    train_idx, val_idx = train_test_split(
        idx_all, test_size=0.2, random_state=replica * 1000
    )

    W_tr = ds["W"][train_idx, :]  # (Ntr, Ngrid)
    y_tr = ds["y"][train_idx]  # (Ntr,)
    C_tr = ds["C"][np.ix_(train_idx, train_idx)]  # (Ntr,Ntr)

    W_val = ds["W"][val_idx, :]  # (Nval, Ngrid)
    y_val = ds["y"][val_idx]  # (Nval,)
    C_val = ds["C"][np.ix_(val_idx, val_idx)]  # (Nval,Nval)

    # train-only dataset for NUTS hyperparams
    ds_tr = dict(ds)
    ds_tr["W"] = W_tr
    ds_tr["y"] = y_tr
    ds_tr["C"] = C_tr

    # ----------------------------
    # Run NUTS on TRAIN only (and include sumrule if implemented in nuts.py)
    # ----------------------------
    idata = sample_hyperparams_nuts(ds_tr, cfg)

    pcfg = cfg.get("gp_prefactor", {})
    pref_mode = str(pcfg.get("mode", "legacy")).lower()  # legacy | prefactor | none

    if pref_mode == "prefactor":
        var_names = ["alpha", "beta", "l0", "sigma"]
    elif pref_mode == "legacy":
        var_names = ["alpha", "l0", "sigma"]
    elif pref_mode == "none":
        var_names = ["l0", "sigma"]
    else:
        raise ValueError(
            f"Unknown gp_prefactor.mode={pref_mode!r} (use legacy|prefactor|none)"
        )

    # --- Prevent ArviZ KeyError if a name isn't in the dataset ---
    present = set(idata.posterior.data_vars)
    var_names = [v for v in var_names if v in present]

    summ = az.summary(idata, var_names=var_names)
    summ.to_csv(os.path.join(out, "summary.csv"))

    # ----------------------------
    # Prepare theta samples
    # ----------------------------
    l0_s = idata.posterior["l0"].values.ravel()
    sigma2_s = idata.posterior["sigma2"].values.ravel()
    print(pref_mode)
    if pref_mode == "prefactor":
        alpha_s = idata.posterior["alpha"].values.ravel()
        beta_s = idata.posterior["beta"].values.ravel()
        theta_samples = np.stack([alpha_s, beta_s, l0_s, sigma2_s], axis=1)  # (S,4)
        alpha_range = (0.0, 5.0)
        print("Prepared theta samples with prefactor alpha and beta.")
    elif pref_mode == "legacy":
        alpha_s = idata.posterior["alpha"].values.ravel()
        theta_samples = np.stack([alpha_s, l0_s, sigma2_s], axis=1)
        alpha_range = (-1.0, 0.0)
    elif pref_mode == "none":
        theta_samples = np.stack([l0_s, sigma2_s], axis=1)
        alpha_range = (-1.0, 0.0)
    else:
        raise ValueError(f"Unknown pref_mode={pref_mode!r}")

    # ----------------------------
    # Fig1
    # ----------------------------
    if cfg.get("eval", {}).get("make_fig1", True):
        fig1 = plot_fig1(idata, alpha_range=alpha_range)
        fig1.savefig(os.path.join(out, "fig1.pdf"))

    # ----------------------------
    # Fig2 / posterior of f*
    # ----------------------------
    if cfg.get("eval", {}).get("make_fig2", False):
        transforms = cfg.get("transforms", {})
        meta = ds.get("meta", {})

        xgrid = np.asarray(ds["xgrid"], float)
        xgrid_t = log_x_gp(xgrid, transforms)

        X_train = xgrid_t.reshape(-1, 1)

        x_train_phys = np.asarray(ds["xgrid"], float).reshape(-1)

        if "xgrid_ext" in meta:
            x_star_phys = np.asarray(meta["xgrid_ext"], float).reshape(-1)
        else:
            x_star_phys = x_train_phys.copy()

        X_star = x_star_phys.reshape(-1, 1)

        # ---- kernel / numeric params ----
        kcfg = cfg.get("kernel", {})
        delta = float(kcfg.get("delta", 1e-5))
        x_floor = float(kcfg.get("x_floor", 1e-12))
        jitter = float(kcfg.get("jitter", 1e-10))
        jitter_star = float(kcfg.get("jitter_star", 1e-10))
        nu = float(kcfg.get("nu", 1.5))

        fstar_kernel = str(cfg.get("kernel", {}).get("name", "gibbs")).lower()
        print(f"Selected f* posterior kernel: {fstar_kernel}")

        max_s = cfg.get("eval", {}).get("fstar_max_samples", None)
        max_s = int(max_s) if max_s is not None else None

        # ----------------------------
        # Sumrule pseudo-observation parameters (match NN)
        # ----------------------------
        kcfg = cfg["kernel"]
        lambda_sr = float(kcfg.get("lambda_sr", 0.0))

        sr_a = sr_ref = sr_tau2 = None
        if lambda_sr > 0.0:
            xt3_true = np.asarray(ds["meta"]["xt3_true"], float).ravel()
            if xt3_true.size != xgrid.size:
                raise ValueError("meta['xt3_true'] must be defined on the full xgrid.")

            # ref = ∫ (xt3_true/x) dx  (same as NN)
            sr_ref = float(np.trapz(xt3_true / xgrid, xgrid))

            # I(f) ≈ sum_i w_i * f_i/x_i = a^T f
            w = _trapz_weights(xgrid)
            sr_a = (w / xgrid).astype(float)

            # tau^2 is the noise variance of the pseudo-observation
            # matching NN penalty: 0.5*(I-ref)^2/tau^2 == lambda_sr*(I-ref)^2
            sr_tau2 = 1.0 / (2.0 * lambda_sr)

        # ----------------------------
        # Sample f* posterior conditioned on TRAIN (+ sumrule)
        # ----------------------------
        replicas, means, vars_f = sample_replicas(
            theta_samples,
            X_train=X_train,
            X_star=X_star,
            FK=W_tr,
            CY=C_tr,
            y=y_tr,
            kernel=fstar_kernel,
            seed=int(cfg.get("seed", 0)),
            delta=delta,
            x_floor=x_floor,
            jitter_cyt=jitter,
            jitter_star=jitter_star,
            nu=nu,
            max_samples=max_s,
            sr_a=sr_a,
            sr_ref=sr_ref,
            sr_tau2=sr_tau2,
            pref_mode=pref_mode,
            x_train_phys=x_train_phys,
            x_star_phys=x_star_phys,
        )

        # ----------------------------
        # Compute VAL χ²/pt (no sumrule penalty) – match NN logging
        # Use posterior mean f for each theta draw
        # ----------------------------
        Cval = 0.5 * (C_val + C_val.T) + jitter * np.eye(C_val.shape[0])
        Lval = np.linalg.cholesky(Cval)

        def solve_Cval(v):
            a = np.linalg.solve(Lval, v)
            return np.linalg.solve(Lval.T, a)

        chi2_pts = []
        x_star_1d = np.asarray(x_star_phys, float).reshape(-1)
        x_fk_1d = np.asarray(x_train_phys, float).reshape(-1)

        chi2_pts = []
        for s in range(means.shape[0]):
            f_mean_star = means[s].reshape(-1)  # length N* (extended)

            # map back to FK grid so W_val @ f makes sense
            f_mean_fk = np.interp(x_fk_1d, x_star_1d, f_mean_star)  # length Ngrid

            y_pred_val = W_val @ f_mean_fk
            r = y_pred_val - y_val
            chi2 = float(r @ solve_Cval(r))
            chi2_pts.append(chi2 / float(len(val_idx)))

        print(
            f"GP Val χ²/pt: mean={np.mean(chi2_pts):.6f}, std={np.std(chi2_pts):.6f}, "
            f"Nval={len(val_idx)}"
        )

        # optional sumrule diagnostic (like NN ΔSR)
        if sr_a is not None:
            I_means = np.array(
                [
                    float(
                        np.trapz(
                            np.interp(x_fk_1d, x_star_1d, means[s]) / x_fk_1d, x_fk_1d
                        )
                    )
                    for s in range(means.shape[0])
                ]
            )
            d = I_means - sr_ref
            print(f"GP ΔSR (mean±std): {d.mean():+.3e} ± {d.std():.3e}")

        # ----------------------------
        # Plot + save
        # ----------------------------
        mean_curve = replicas.mean(axis=0)
        lo68, hi68 = np.percentile(replicas, [16, 84], axis=0)
        lo95, hi95 = np.percentile(replicas, [2.5, 97.5], axis=0)

        xt3_true_star = None
        if (
            "xgrid_ext" in meta
            and "xt3_ext" in meta
            and x_star_phys.size == np.asarray(meta["xt3_ext"]).size
        ):
            xt3_true_star = np.asarray(meta["xt3_ext"], float).reshape(-1)
            print("GP: using xt3_ext truth on xgrid_ext.")
        elif "xt3_true" in meta:
            xt3_true = np.asarray(meta["xt3_true"], float).reshape(-1)
            # fallback: interpolate FK-grid truth to x_star
            xt3_true_star = np.interp(x_star_phys, xgrid, xt3_true)
            print("GP: using interpolated xt3_true truth.")

        plot_fig2(
            x_star_phys,
            mean_curve,
            lo68,
            hi68,
            lo95,
            hi95,
            xt3_true_star=xt3_true_star,
            outpath=os.path.join(out, "fig2_gp.pdf"),
        )

        uq = gp_uq_bands(means, vars_f, sigma2_obs_star=0.0, draws_per_theta=1, seed=0)
        mean_curve_uq = uq["mean_curve"]
        lo68_u, hi68_u, lo95_u, hi95_u = uq["bands_f_mm"]
        plot_fig2(
            x_star_phys,
            mean_curve_uq,
            lo68_u,
            hi68_u,
            lo95_u,
            hi95_u,
            xt3_true_star=xt3_true_star,
            outpath=os.path.join(out, "fig2_gp_uncbands.pdf"),
        )


        x_star_1d = np.asarray(x_star_phys, float).reshape(-1)              # (N*,)
        x_fk_1d = np.asarray(x_train_phys, float).reshape(-1)               # (Nfk,)

        y_pred_members = np.empty((replicas.shape[0], n_data), dtype=np.float64)

        for s in range(replicas.shape[0]):
            f_star = replicas[s].reshape(-1)                              # (N*,)
            f_fk = np.interp(x_fk_1d, x_star_1d, f_star)                     # (Nfk,)
            y_pred_members[s] = W_full @ f_fk                                # (Ndat,)

        y_pred_mean = y_pred_members.mean(axis=0)
        sigma_ens2 = (
            y_pred_members.var(axis=0, ddof=1)
            if replicas.shape[0] > 1
            else np.zeros_like(y_pred_mean)
        )

        # Observable/data-space covariance (legacy / diagnostics)
        if y_pred_members.shape[0] > 1:
            C_ens = np.cov(y_pred_members, rowvar=False, ddof=1)
        else:
            C_ens = np.zeros((n_data, n_data), dtype=np.float64)
        C_ens = ensure_symmetric(C_ens)

        # x-space covariance for downstream HERA chi2
        if replicas.shape[0] > 1:
            cov_ens_f = np.cov(replicas, rowvar=False, ddof=1)
        else:
            cov_ens_f = np.zeros((replicas.shape[1], replicas.shape[1]), dtype=np.float64)
        cov_ens_f = ensure_symmetric(cov_ens_f)
        xgrid_cov_ens_f = np.asarray(x_star_phys, dtype=np.float64).ravel()

        # Optional total x-space covariance using posterior vars_f
        if vars_f is not None and np.size(vars_f) > 0:
            var_het_f = np.mean(np.asarray(vars_f, dtype=np.float64), axis=0)
        else:
            var_het_f = np.zeros(cov_ens_f.shape[0], dtype=np.float64)

        cov_het_f = np.diag(var_het_f)
        cov_tot_f = cov_ens_f + cov_het_f

        sigma2_xg = sigma_ens2 + diagC**2
        sigma_xg = np.sqrt(np.maximum(sigma2_xg, 1e-18))


        save_replicas = bool(cfg.get("eval", {}).get("save_fstar_replicas", True))
        npz_path = os.path.join(out, "gp_summary.npz")
        np.savez(
            npz_path,
            x_star=x_star_phys,
            mean_curve=mean_curve,
            lo68=lo68, hi68=hi68,
            lo95=lo95, hi95=hi95,
            xt3_true_star=xt3_true_star if xt3_true_star is not None else np.array([]),

            # chi2-compatible x-space covariance
            cov_ens_f=cov_ens_f,
            xgrid_cov_ens_f=xgrid_cov_ens_f,
            cov_het_f=cov_het_f,
            cov_tot_f=cov_tot_f,

            # observable/data-space diagnostics
            y_target=y_pseudo,
            y_pred_mean=y_pred_mean,
            sigma_ens2=sigma_ens2,
            diagC=diagC,
            ensC=C_ens,
            sigma_xg=sigma_xg,
            y_pred_members=y_pred_members,

            replicas=replicas if save_replicas else np.array([]),
            means=means if save_replicas else np.array([]),
            vars_f=vars_f if save_replicas else np.array([]),
            train_idx=train_idx,
            val_idx=val_idx,
        )
        print(f"Saved f* posterior summary to {npz_path}")

    print(f"Done. Outputs in {out}")
    return idata
