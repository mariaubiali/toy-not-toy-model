from __future__ import annotations

import os
import numpy as np

from dataloader import load_dataset
from plotting import plot_fig2, plot_fig2_unc
from models.nn_train import train_nn_forward  # single model trainer


def _train_ensemble(ds: dict, cfg: dict):
    ens = cfg.get("nn", {}).get("ensemble", {})
    enabled = bool(ens.get("enabled", True))

    loss_name = str(cfg.get("loss", {}).get("name", "weighted_mse")).lower()
    want_het = loss_name.startswith("het") or "het" in loss_name

    def _get_member_out(cfg_i: dict):
        out_i = train_nn_forward(ds, cfg_i)
        x_star_i = out_i["x_star"]
        mu_i = out_i["f_star_mean"]
        var_i = out_i.get("f_star_var", None)  # should be (N*,) if het enabled
        if (var_i is None) and want_het:
            # If you haven't implemented het outputs yet, we keep running but treat het var as zero.
            var_i = None
        return x_star_i, mu_i, var_i

    if not enabled:
        x_star, mu, var = _get_member_out(cfg)
        mus = mu[None, :]  # (1, N*)
        vars_ = None if var is None else var[None, :]
        return x_star, mus, vars_

    n_members = int(ens.get("n_members", 20))
    seed_offset = int(ens.get("seed_offset", 1000))
    base_seed = int(cfg.get("seed", 0))

    mus_list = []
    vars_list = []
    x_star = None

    for i in range(n_members):
        cfg_i = dict(cfg)
        cfg_i["seed"] = base_seed + seed_offset + i
        x_star_i, mu_i, var_i = _get_member_out(cfg_i)

        x_star = x_star_i
        mus_list.append(mu_i)

        if var_i is not None:
            vars_list.append(var_i)

    mus = np.stack(mus_list, axis=0)  # (S, N*)

    # Only return vars if we actually collected them for all members
    vars_ = None
    if len(vars_list) == len(mus_list) and len(vars_list) > 0:
        vars_ = np.stack(vars_list, axis=0)  # (S, N*)

    return x_star, mus, vars_


def run_from_config(cfg: dict):
    out = cfg.get("output_dir", "outputs/run")
    os.makedirs(out, exist_ok=True)

    ds = load_dataset(cfg["data"])

    # ----------------------------
    # Train NN ensemble -> replicas
    # ----------------------------
    x_star, mus, vars_ = _train_ensemble(ds, cfg)

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

    # If you want quantile bands from ensemble only (epistemic), keep these:
    lo68_ens_q, hi68_ens_q = np.percentile(mus, [16.0, 84.0], axis=0)
    lo95_ens_q, hi95_ens_q = np.percentile(mus, [2.5, 97.5], axis=0)

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

    # ----------------------------
    # True curve (NNPDF)
    # ----------------------------
    xt3_true_star = None
    meta = ds.get("meta", {})
    if "xt3_true" in meta:
        xt3_true = np.asarray(meta["xt3_true"], float).ravel()
        xt3_true_star = np.interp(x_star, ds["xgrid"], xt3_true)

    # ----------------------------
    # Save everything needed for later bias/coverage scripts
    # ----------------------------
    save_members = bool(cfg.get("nn", {}).get("ensemble", {}).get("save_member_preds", True))

    np.savez(
        os.path.join(out, "nn_uncertainty_summary.npz"),
        x_star=x_star,
        mean_curve=mean_curve,
        xt3_true_star=xt3_true_star if xt3_true_star is not None else np.array([]),

        # variances
        var_ens=var_ens,
        var_het=var_het,
        var_tot=var_tot,

        # total bands
        lo68_tot=lo68_tot, hi68_tot=hi68_tot,
        lo95_tot=lo95_tot, hi95_tot=hi95_tot,

        # ens bands (gaussian)
        lo68_ens=lo68_ens, hi68_ens=hi68_ens,
        lo95_ens=lo95_ens, hi95_ens=hi95_ens,

        # het bands (gaussian)
        lo68_het=lo68_het, hi68_het=hi68_het,
        lo95_het=lo95_het, hi95_het=hi95_het,

        # ensemble-only quantile bands (optional, useful sanity check)
        lo68_ens_q=lo68_ens_q, hi68_ens_q=hi68_ens_q,
        lo95_ens_q=lo95_ens_q, hi95_ens_q=hi95_ens_q,

        # optional heavy arrays
        mus=mus if save_members else np.array([]),
        vars=vars_ if (save_members and vars_ is not None) else np.array([]),
    )

    # ----------------------------
    # Plot fig2
    # ----------------------------
    if cfg.get("eval", {}).get("make_fig2", True):
        # By default: plot TOTAL uncertainty if het is enabled, otherwise plot ensemble-only quantiles.
        loss_name = str(cfg.get("loss", {}).get("name", "weighted_mse")).lower()
        het_enabled = (vars_ is not None) and ("het" in loss_name)

        if het_enabled:
            bands_total = (lo68_tot, hi68_tot, lo95_tot, hi95_tot)

            # ensemble uncertainty: use quantiles from mus
            bands_ens = (lo68_ens_q, hi68_ens_q, lo95_ens_q, hi95_ens_q)

            # heteroscedastic uncertainty: gaussian from var_het
            bands_het = (lo68_het, hi68_het, lo95_het, hi95_het)

            plot_fig2_unc(
                x_star=x_star,
                mean_curve=mean_curve,
                bands_total=bands_total,
                bands_ens=bands_ens,
                bands_het=bands_het,
                xt3_true_star=xt3_true_star,
                outpath=os.path.join(out, "fig2_nn_unc.pdf"),
            )
        else:
            plot_fig2(
                x_star=x_star,
                mean_curve=mean_curve,
                lo68=lo68_ens_q, hi68=hi68_ens_q,
                lo95=lo95_ens_q, hi95=hi95_ens_q,
                xt3_true_star=xt3_true_star,
                outpath=os.path.join(out, "fig2_nn.pdf"),
            )

    print(f"Done. NN outputs in {out}")
    return {
        "x_star": x_star,
        "mean_curve": mean_curve,
        "var_ens": var_ens,
        "var_het": var_het,
        "var_tot": var_tot,
        "xt3_true_star": xt3_true_star,
    }