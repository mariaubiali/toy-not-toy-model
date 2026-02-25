from __future__ import annotations

import os
import numpy as np

from dataloader import load_dataset
from plotting import plot_fig2, plot_fig2_unc, select_truth_and_band
from models.nn_train import train_nn_forward, train_nn_ensemble_forward
from models.ntk import run_ntk


def _train_ensemble(ds: dict, cfg: dict):
    
    model_type = str(cfg.get("model", {}).get("type", "nn")).lower()
    if model_type == "ntk":
        out = run_ntk(ds, cfg)
        x_star = out["xgrid"]
        mus = out["f_grid_mean"][None, :]
        vars_ = out.get("f_grid_var", None)
        vars_ = None if vars_ is None else vars_[None, :]
        return x_star, mus, vars_, out
    
    ens = cfg.get("nn", {}).get("ensemble", {})
    enabled = bool(ens.get("enabled", False))

    if not enabled:
        out = train_nn_forward(ds, cfg)
        x_star = out["xgrid"]
        mus = out["f_grid_mean"][None, :]
        vars_ = out.get("f_grid_var", None)
        vars_ = None if vars_ is None else vars_[None, :]
        return x_star, mus, vars_, None

    out = train_nn_ensemble_forward(ds, cfg)
    x_star = out["xgrid"]
    mus = out["replicas"]
    vars_ = out.get("vars_replicas", None)
    return x_star, mus, vars_, None


def run_from_config(cfg: dict):
    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    ds = load_dataset(cfg["data"])
    meta = ds.get("meta", {})
    W_full = np.asarray(ds["W"], dtype=np.float64)         # (Ndat, Ngrid)
    y_pseudo = np.asarray(ds['y'], dtype=np.float64).ravel()
    C_full = np.asarray(ds["C"], dtype=np.float64) 
    diagC = np.diag(C_full)
    x_fk  = np.asarray(ds["xgrid"], dtype=np.float64).ravel()  

    # ----------------------------
    # Train NN ensemble -> replicas
    # ----------------------------
    xgrid, mus, vars_, out_ntk = _train_ensemble(ds, cfg)

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

    # If you still want the diagonal variance too:
    sigma2_ens_xg = np.diag(C_ens)
    # sigma2_ens_xg = y_pred_members.var(axis=0, ddof=1) if y_pred_members.shape[0] > 1 else np.zeros_like(y_pred_mean)

    # Your sigma definition:
    sigma2 = sigma2_ens_xg + diagC**2
    sigma_xg = np.sqrt(np.maximum(sigma2, 1e-18))

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

    # total
    var_tot = var_ens + var_het

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

    model_type = str(cfg.get("model", {}).get("type", "nn")).lower()
    if model_type == "ntk" and out_ntk is not None:
        print("plot Ntk error")
        # If run_ntk stored bands, prefer them
        if "f_grid_lo68" in out_ntk and "f_grid_hi68" in out_ntk:
            lo68_tot = out_ntk["f_grid_lo68"]
            hi68_tot = out_ntk["f_grid_hi68"]
        if "f_grid_lo95" in out_ntk and "f_grid_hi95" in out_ntk:
            lo95_tot = out_ntk["f_grid_lo95"]
            hi95_tot = out_ntk["f_grid_hi95"]

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

    np.savez(
        os.path.join(out, "nn_summary.npz"),
        xgrid=xgrid,
        mean_curve=mean_curve,
        xt3_true_star=xt3_true if xt3_true is not None else np.array([]),
        true_sigma=true_sigma if true_sigma is not None else np.array([]),
        var_ens=var_ens,
        var_het=var_het,
        # Data space (for pull)
        y_pseudo=y_pseudo,
        y_pred_mean=y_pred_mean,
        sigma2_ens_xg=sigma2_ens_xg,
        diagC=diagC,
        ensC=C_ens,
        sigma_xg=sigma_xg,
        mus=mus if save_members else np.array([]),
        vars=vars_ if (save_members and vars_ is not None) else np.array([]),
    )

    # ----------------------------
    # Plot fig2
    # ----------------------------
    if cfg.get("eval", {}).get("make_fig2", True):
        output_cfg = cfg.get("nn", {}).get("out_dim", {})
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
