from __future__ import annotations

import os
import numpy as np

from dataloader import load_dataset
from plotting import plot_fig2, plot_fig2_unc, select_truth_and_band
from models.nn_train import train_nn_forward, train_nn_ensemble_forward, train_l2_replicas_forward
from models.ntk import run_ntk


def _train_ensemble(ds: dict, cfg: dict):
    
    model_type = str(cfg.get("model", {}).get("type", "nn")).lower()
    if model_type == "ntk":
        out = run_ntk(ds, cfg)
        x_star = np.asarray(out["xgrid"], dtype=np.float64)
        mus = np.asarray(out["mean_curve"], dtype=np.float64)[None, :]
        vars_ = np.asarray(out.get("var_ens", np.zeros_like(mus[0])), dtype=np.float64)[None, :]
        return x_star, mus, vars_, out
    
    ens = cfg.get("nn", {}).get("ensemble", {})
    enabled = bool(ens.get("enabled", False))

    y_loaded = np.asarray(ds["y"])
    has_l2_replicas = y_loaded.ndim == 2

    if has_l2_replicas:
        out = train_l2_replicas_forward(ds, cfg)
        x_star = out["xgrid"]
        mus = out["replicas"]
        vars_ = out.get("vars_replicas", None)
        return x_star, mus, vars_, out
    elif enabled:
        out = train_nn_ensemble_forward(ds, cfg)
        x_star = out["xgrid"]
        mus = out["replicas"]
        vars_ = out.get("vars_replicas", None)
        return x_star, mus, vars_, out

    out = train_nn_forward(ds, cfg)
    x_star = out["xgrid"]
    mus = out["f_grid_mean"][None, :]
    vars_ = out.get("f_grid_var", None)
    vars_ = None if vars_ is None else vars_[None, :]

    return x_star, mus, vars_, out

def ensure_symmetric(C):
    C = np.asarray(C, dtype=np.float64)
    return 0.5 * (C + C.T)


def _collect_ntk_stage_summary(out_ntk: dict, stage: str) -> dict:
    """Map staged NTK outputs onto the same label schema used by nn_summary.npz."""
    prefix = f"{stage}_ntk_"
    mean = out_ntk.get(f"{stage}_ntk_f_mean", None)
    if mean is None:
        return {}

    block = {
        f"{prefix}xgrid": np.asarray(out_ntk[f"{stage}_ntk_xgrid"], dtype=np.float64),
        f"{prefix}mean_curve": np.asarray(mean, dtype=np.float64),
        f"{prefix}var_ens": np.asarray(out_ntk.get(f"{stage}_ntk_f_var", np.zeros_like(mean)), dtype=np.float64),
        f"{prefix}var_het": np.zeros_like(np.asarray(mean, dtype=np.float64)),
        f"{prefix}lo68": np.asarray(out_ntk[f"{stage}_ntk_f_lo68"], dtype=np.float64),
        f"{prefix}hi68": np.asarray(out_ntk[f"{stage}_ntk_f_hi68"], dtype=np.float64),
        f"{prefix}lo95": np.asarray(out_ntk[f"{stage}_ntk_f_lo95"], dtype=np.float64),
        f"{prefix}hi95": np.asarray(out_ntk[f"{stage}_ntk_f_hi95"], dtype=np.float64),
        f"{prefix}sigma_f": np.asarray(out_ntk.get(f"{stage}_ntk_f_std", np.zeros_like(mean)), dtype=np.float64),
    }

    f_cov = out_ntk.get(f"{stage}_ntk_f_cov", None)
    xgrid = block[f"{prefix}xgrid"]
    if f_cov is not None:
        cov = ensure_symmetric(np.asarray(f_cov, dtype=np.float64))
    else:
        var = block[f"{prefix}var_ens"]
        cov = np.diag(np.asarray(var, dtype=np.float64))
    block[f"{prefix}cov_ens_f"] = cov
    block[f"{prefix}cov_het_f"] = np.zeros_like(cov)
    block[f"{prefix}cov_tot_f"] = cov.copy()
    block[f"{prefix}xgrid_cov_ens_f"] = xgrid.copy()
    return block


def _promote_post_ntk_to_primary(save_dict: dict, out_ntk: dict) -> None:
    """Overwrite primary summary keys with post-training NTK GP outputs."""
    block = _collect_ntk_stage_summary(out_ntk, "post")
    if not block:
        return

    save_dict["xgrid"] = block["post_ntk_xgrid"]
    save_dict["mean_curve"] = block["post_ntk_mean_curve"]
    save_dict["var_ens"] = block["post_ntk_var_ens"]
    save_dict["var_het"] = block["post_ntk_var_het"]
    save_dict["lo68"] = block["post_ntk_lo68"]
    save_dict["hi68"] = block["post_ntk_hi68"]
    save_dict["lo95"] = block["post_ntk_lo95"]
    save_dict["hi95"] = block["post_ntk_hi95"]
    save_dict["cov_ens_f"] = block["post_ntk_cov_ens_f"]
    save_dict["xgrid_cov_ens_f"] = block["post_ntk_xgrid_cov_ens_f"]
    save_dict["cov_het_f"] = block["post_ntk_cov_het_f"]
    save_dict["cov_tot_f"] = block["post_ntk_cov_tot_f"]
    save_dict["sigma_f"] = block["post_ntk_sigma_f"]


def run_from_config(cfg: dict):
    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    ds = load_dataset(cfg["data"])
    meta = ds.get("meta", {})
    model_type = str(cfg.get("model", {}).get("type", "nn")).lower()
    ntk_when = str(cfg.get("ntk", {}).get("when", "none")).lower()
    W_full = np.asarray(ds["W"], dtype=np.float64)         # (Ndat, Ngrid)
    y_loaded = np.asarray(ds["y"], dtype=np.float64)
    if y_loaded.ndim == 1:
        y_pseudo = y_loaded
    else:
        y_pseudo = y_loaded.mean(axis=1)
    C_full = np.asarray(ds["C"], dtype=np.float64) 
    diagC = np.diag(C_full)
    x_fk  = np.asarray(ds["xgrid"], dtype=np.float64).ravel()  

    # ----------------------------
    # Train NN ensemble -> replicas
    # ----------------------------
    xgrid, mus, vars_, out_ntk = _train_ensemble(ds, cfg)
    loss_name = str(cfg.get("loss", {}).get("name", "weighted_mse")).lower()
    out_dim = int(cfg.get("nn", {}).get("out_dim", 1))
    het_enabled_cfg = (loss_name in {"mse_het", "chi_het"}) and (out_dim == 2)
    if not het_enabled_cfg and not (model_type == "nn" and ntk_when == "post"):
        vars_ = None

    mus_fk = np.empty((mus.shape[0], x_fk.size), dtype=np.float64)
    for s in range(mus.shape[0]):
        mus_fk[s] = np.interp(x_fk, xgrid, mus[s])

    # --- Push to data space on FK grid ---
    y_pred_members = mus_fk @ W_full.T                               # (S, Ndat)
    y_pred_mean = y_pred_members.mean(axis=0)

    if y_pred_members.shape[0] > 1:
        C_ens = np.cov(y_pred_members, rowvar=False, ddof=1)  # (Ndat, Ndat)
    else:
        C_ens = np.zeros((y_pred_members.shape[1], y_pred_members.shape[1]))

    # ---------- NTK OVERRIDE: use GP predictive covariance ----------
    if model_type == "ntk" and out_ntk is not None and "f_cov" in out_ntk:
        print("in NTK override")

        f_cov = np.asarray(out_ntk["f_cov"], dtype=np.float64)
        x_src = np.asarray(out_ntk.get("x_pred_used", xgrid), dtype=np.float64).ravel()

        A_fk = interp_matrix_1d(x_src, x_fk)
        f_cov_fk = A_fk @ f_cov @ A_fk.T

        # Symmetrize after interpolation
        f_cov_fk = 0.5 * (f_cov_fk + f_cov_fk.T)

        C_ens = W_full @ f_cov_fk @ W_full.T
        C_ens = 0.5 * (C_ens + C_ens.T)

        # ---- Make SPD for downstream Cholesky inversions ----
        # 1) PSD projection (clips small negative eigenvalues from numerics)
        evals, evecs = np.linalg.eigh(C_ens)
        evals = np.clip(evals, 0.0, None)
        C_ens = (evecs * evals) @ evecs.T
        C_ens = 0.5 * (C_ens + C_ens.T)

        # 2) Add a small jitter scaled to the typical variance level
        # (scale makes this robust across datasets)
        diag_mean = float(np.mean(np.diag(C_ens))) if C_ens.size else 1.0
        jitter = 1e-8 * max(diag_mean, 1.0)   # you can tune 1e-8 -> 1e-6 if needed
        C_ens.flat[::C_ens.shape[0] + 1] += jitter


    sigma2_ens_xg = np.diag(C_ens)

    # Build heteroscedastic covariance in data-space:
    # C_het = E_s[ W diag(var_f_s_fk) W^T ].
    if vars_ is None:
        C_het = np.zeros_like(C_ens)
    else:
        vars_fk = np.empty((vars_.shape[0], x_fk.size), dtype=np.float64)
        for s in range(vars_.shape[0]):
            vars_fk[s] = np.interp(x_fk, xgrid, vars_[s])
        C_het = np.zeros_like(C_ens)
        for s in range(vars_fk.shape[0]):
            C_het += (W_full * vars_fk[s][None, :]) @ W_full.T
        C_het /= float(vars_fk.shape[0])
        C_het = 0.5 * (C_het + C_het.T)

    sigma2_het_xg = np.diag(C_het)
    C_tot = C_ens + C_het + diagC
    C_tot = 0.5 * (C_tot + C_tot.T)
    sigma2 = np.diag(C_tot)
    sigma_xg = np.sqrt(np.maximum(sigma2, 1e-18))

    # print("C_het: ", C_het.shape, C_het)

    # ----------------------------
    # Bands + mean from ensemble
    # ----------------------------
    mean_curve = mus.mean(axis=0)  # (N*,)

    # epistemic (ensemble) variance
    var_ens = mus.var(axis=0, ddof=1) if mus.shape[0] > 1 else np.zeros_like(mean_curve)

    # aleatoric (heteroscedastic) variance on f*(x), if provided
    if vars_ is None:
        var_het = np.zeros_like(mean_curve)
    else:
        var_het = vars_.mean(axis=0)

    var_tot = var_ens + var_het
    # cov_ens_f is the x-space covariance to be used later for chi2.
    # NN: empirical covariance of trained ensemble replicas on xgrid.
    # NTK: GP/NTK predictive covariance at initialization.
    cov_ens_f = (
        np.cov(mus, rowvar=False, ddof=1)
        if mus.shape[0] > 1
        else np.zeros((mus.shape[1], mus.shape[1]), dtype=np.float64)
    )
    cov_ens_f = ensure_symmetric(cov_ens_f)
    xgrid_cov_ens_f = np.asarray(xgrid, dtype=np.float64).ravel()

    if model_type == "ntk" and out_ntk is not None and "f_cov" in out_ntk:
        print("Using NTK f_cov as cov_ens_f for chi2.")
        cov_ens_f = np.asarray(out_ntk["f_cov"], dtype=np.float64)
        cov_ens_f = ensure_symmetric(cov_ens_f)
        xgrid_cov_ens_f = np.asarray(out_ntk.get("x_pred_used", xgrid), dtype=np.float64).ravel()

    cov_het_f = np.diag(var_het)
    cov_tot_f = cov_ens_f + cov_het_f
    sigma_f = np.sqrt(np.maximum(np.diag(cov_tot_f), 1e-18))

    # Gaussian-approx bands for total/ens/het (simple & fast)
    std_ens = np.sqrt(np.maximum(var_ens, 0.0))
    std_het = np.sqrt(np.maximum(var_het, 0.0))
    std_tot = np.sqrt(np.maximum(var_tot, 0.0))

    lo68_tot, hi68_tot = mean_curve - 1.0 * std_tot, mean_curve + 1.0 * std_tot
    lo95_tot, hi95_tot = mean_curve - 1.96 * std_tot, mean_curve + 1.96 * std_tot

    lo68_ens, hi68_ens = mean_curve - 1.0 * std_ens, mean_curve + 1.0 * std_ens
    lo95_ens, hi95_ens = mean_curve - 1.96 * std_ens, mean_curve + 1.96 * std_ens

    lo68_het, hi68_het = mean_curve - 1.0 * std_het, mean_curve + 1.0 * std_het
    lo95_het, hi95_het = mean_curve - 1.96 * std_het, mean_curve + 1.96 * std_het

    if model_type == "ntk" and out_ntk is not None:
        print("plot NTK error")
        # Prefer primary summary keys if present on the standalone NTK route.
        if "lo68" in out_ntk and "hi68" in out_ntk:
            lo68_tot = np.asarray(out_ntk["lo68"], dtype=np.float64)
            hi68_tot = np.asarray(out_ntk["hi68"], dtype=np.float64)
        elif "f_grid_lo68" in out_ntk and "f_grid_hi68" in out_ntk:
            lo68_tot = np.asarray(out_ntk["f_grid_lo68"], dtype=np.float64)
            hi68_tot = np.asarray(out_ntk["f_grid_hi68"], dtype=np.float64)
        if "lo95" in out_ntk and "hi95" in out_ntk:
            lo95_tot = np.asarray(out_ntk["lo95"], dtype=np.float64)
            hi95_tot = np.asarray(out_ntk["hi95"], dtype=np.float64)
        elif "f_grid_lo95" in out_ntk and "f_grid_hi95" in out_ntk:
            lo95_tot = np.asarray(out_ntk["f_grid_lo95"], dtype=np.float64)
            hi95_tot = np.asarray(out_ntk["f_grid_hi95"], dtype=np.float64)

    # ----------------------------
    # True curve (NNPDF)
    # ----------------------------
    meta = ds.get("meta", {})
    xt3_true = None

    # If we trained/evaluated on xgrid_eval, prefer the matching truth
    if "xgrid_ext" in meta and "xt3_ext" in meta:
        xt3_true = np.asarray(meta["xt3_ext"], float).ravel()
        print("Using extended xgrid truth xt3_ext for evaluation.")
    elif "xt3_true" in meta:
        xt3_true = np.asarray(meta["xt3_true"], float).ravel()

    x_plot, xt3_true_plot, true_sigma = select_truth_and_band(ds, x_model=xgrid, lam=None)

    # ----------------------------
    # Save everything needed for later bias/coverage scripts
    # ----------------------------
    save_members = bool(
        cfg.get("nn", {}).get("ensemble", {}).get("save_member_preds", True)
    )
    print("save results")
    save_dict = {
        "xgrid": xgrid,
        "mean_curve": mean_curve,
        "lo68": lo68_tot,
        "hi68": hi68_tot,
        "lo95": lo95_tot,
        "hi95": hi95_tot,
        "xt3_true_star": xt3_true if xt3_true is not None else np.array([]),
        "true_sigma": true_sigma if true_sigma is not None else np.array([]),
        "var_ens": var_ens,
        "var_het": var_het,
        # Data space (for pull)
        "y_target": y_pseudo,
        "y_pred_mean": y_pred_mean,
        "sigma2_ens_xg": sigma2_ens_xg,
        "sigma2_het_xg": sigma2_het_xg,
        "diagC": diagC,
        "ensC": C_ens,
        "hetC": C_het,
        "totalC": C_tot,
        "sigma_xg": sigma_xg,
        "cov_ens_f": cov_ens_f,
        "xgrid_cov_ens_f": xgrid_cov_ens_f,
        "cov_het_f": cov_het_f,
        "cov_tot_f": cov_tot_f,
        "sigma_f": sigma_f,
        "y_pred_members": y_pred_members,
        "mus": mus if save_members else np.array([]),
        "vars": vars_ if (save_members and vars_ is not None) else np.array([]),
    }

    if out_ntk is not None:
        save_dict.update(_collect_ntk_stage_summary(out_ntk, "init"))
        save_dict.update(_collect_ntk_stage_summary(out_ntk, "post"))

    if model_type == "nn" and ntk_when == "post" and out_ntk is not None:
        _promote_post_ntk_to_primary(save_dict, out_ntk)

    np.savez(os.path.join(out, "nn_summary.npz"), **save_dict)

    # Promote post-NTK to the primary plotting variables for the trained-NN route.
    if model_type == "nn" and ntk_when == "post" and out_ntk is not None:
        post_block = _collect_ntk_stage_summary(out_ntk, "post")
        if post_block:
            xgrid = np.asarray(post_block["post_ntk_xgrid"], dtype=np.float64)
            mean_curve = np.asarray(post_block["post_ntk_mean_curve"], dtype=np.float64)
            lo68_tot = np.asarray(post_block["post_ntk_lo68"], dtype=np.float64)
            hi68_tot = np.asarray(post_block["post_ntk_hi68"], dtype=np.float64)
            lo95_tot = np.asarray(post_block["post_ntk_lo95"], dtype=np.float64)
            hi95_tot = np.asarray(post_block["post_ntk_hi95"], dtype=np.float64)

    # For the standalone NTK route, prefer the direct summary keys if available.
    if model_type == "ntk" and out_ntk is not None:
        xgrid = np.asarray(out_ntk.get("xgrid", xgrid), dtype=np.float64)
        mean_curve = np.asarray(out_ntk.get("mean_curve", mean_curve), dtype=np.float64)
        lo68_tot = np.asarray(out_ntk.get("lo68", lo68_tot), dtype=np.float64)
        hi68_tot = np.asarray(out_ntk.get("hi68", hi68_tot), dtype=np.float64)
        lo95_tot = np.asarray(out_ntk.get("lo95", lo95_tot), dtype=np.float64)
        hi95_tot = np.asarray(out_ntk.get("hi95", hi95_tot), dtype=np.float64)

    # ----------------------------
    # Plot fig2
    # ----------------------------
    if cfg.get("eval", {}).get("make_fig2", True):
        het_enabled = (vars_ is not None)

        if het_enabled:
            print("Plotting NN fig2 with heteroscedastic uncertainty bands.")
            bands_total = (lo68_tot, hi68_tot, lo95_tot, hi95_tot)

            # ensemble uncertainty: use quantiles from mus
            bands_ens = (lo68_ens, hi68_ens, lo95_ens, hi95_ens)

            # heteroscedastic uncertainty: gaussian from var_het
            bands_het = (lo68_het, hi68_het, lo95_het, hi95_het)

            plot_fig2_unc(
                x_star=xgrid,
                mean_curve=mean_curve,
                bands_total=bands_total,
                bands_ens=bands_ens,
                bands_het=bands_het,
                xt3_true_star=xt3_true,
                true_sigma=true_sigma,
                outpath=os.path.join(out, "fig2_nn_unc.pdf"),
            )

            plot_fig2(
                x_star=xgrid,
                mean_curve=mean_curve,
                lo68=lo68_tot,
                hi68=hi68_tot,
                lo95=lo95_tot,
                hi95=hi95_tot,
                xt3_true_star=xt3_true,
                true_sigma=true_sigma,
                outpath=os.path.join(out, "fig2_nn.pdf"),
            )
        else:
            plot_fig2(
                x_star=xgrid,
                mean_curve=mean_curve,
                lo68=lo68_tot,
                hi68=hi68_tot,
                lo95=lo95_tot,
                hi95=hi95_tot,
                xt3_true_star=xt3_true,
                true_sigma=true_sigma,
                outpath=os.path.join(out, "fig2_nn.pdf"),
            )

    print(f"Done. NN outputs in {out}")
    return {
        "xgrid": xgrid,
        "mean_curve": mean_curve,
        "var_ens": var_ens,
        "var_het": var_het,
        "var_tot": var_tot,
        "xt3_true": xt3_true,
        "true_sigma": true_sigma,
    }


def interp_matrix_1d(x_src, x_tgt):
    """
    Build A such that f(x_tgt) ≈ A f(x_src) using piecewise-linear interpolation.
    x_src must be sorted ascending. Assumes x_tgt within [min(x_src), max(x_src)].
    """
    x_src = np.asarray(x_src, dtype=np.float64).ravel()
    x_tgt = np.asarray(x_tgt, dtype=np.float64).ravel()
    n_src = x_src.size
    n_tgt = x_tgt.size

    A = np.zeros((n_tgt, n_src), dtype=np.float64)

    # indices i such that x_src[i-1] <= x < x_src[i]
    idx = np.searchsorted(x_src, x_tgt, side="left")
    idx = np.clip(idx, 1, n_src - 1)

    x0 = x_src[idx - 1]
    x1 = x_src[idx]
    t = (x_tgt - x0) / (x1 - x0 + 1e-300)

    A[np.arange(n_tgt), idx - 1] = (1.0 - t)
    A[np.arange(n_tgt), idx]     = t
    return A
