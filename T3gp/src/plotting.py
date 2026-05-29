from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})


def _interp_matrix_linear(x_src, x_tgt):
    x_src = np.asarray(x_src, float)
    x_tgt = np.asarray(x_tgt, float)

    N, M = x_src.size, x_tgt.size
    P = np.zeros((M, N), dtype=float)

    idx = np.searchsorted(x_src, x_tgt, side="right") - 1
    idx = np.clip(idx, 0, N - 2)

    x0 = x_src[idx]
    x1 = x_src[idx + 1]
    t = (x_tgt - x0) / np.maximum(x1 - x0, 1e-300)

    rows = np.arange(M)
    P[rows, idx] = 1.0 - t
    P[rows, idx + 1] = t

    return P


def _truth_cov_fk(W, Cy, lam=None):
    L = np.linalg.cholesky(Cy)

    Y = np.linalg.solve(L, W)
    Cy_inv_W = np.linalg.solve(L.T, Y)

    A = W.T @ Cy_inv_W

    Nx = A.shape[0]
    if lam is None:
        lam = 1e-1 * (np.trace(A) / max(Nx, 1))

    M = A + lam * np.eye(Nx)

    # Covariance of ridge-GLS estimator:
    # Cf = M^{-1} A M^{-1}
    X = np.linalg.solve(M, A)
    Cf = X @ np.linalg.solve(M, np.eye(Nx))

    return Cf, lam


def select_truth_and_band(ds: dict, x_model: np.ndarray, lam: float | None = None):
    """
    Returns:
      x_plot: grid to plot on, extended if available
      xt3_true_plot: truth on x_plot, or None
      sigma_true_plot: 1-sigma truth band on x_plot from W and C, or None
    """
    meta = ds.get("meta", {})

    x_fk = np.asarray(ds["xgrid"], float).ravel()
    W = np.asarray(ds["W"], float)
    Cy = np.asarray(ds["C"], float)

    if "xgrid_ext" in meta and "xt3_ext" in meta:
        x_plot = np.asarray(meta["xgrid_ext"], float).ravel()
        xt3_true_plot = np.asarray(meta["xt3_ext"], float).ravel()
        print("Using extended xgrid")
    else:
        x_plot = np.asarray(x_model, float).ravel()
        xt3_true_plot = (
            np.asarray(meta["xt3_true"], float).ravel()
            if "xt3_true" in meta
            else None
        )

    if xt3_true_plot is None:
        return x_plot, None, None

    Cf_fk, _lam = _truth_cov_fk(W, Cy, lam=lam)

    if x_plot.shape == x_fk.shape and np.allclose(x_plot, x_fk, rtol=0, atol=0):
        sigma = np.sqrt(np.clip(np.diag(Cf_fk), 0.0, None))
        return x_plot, xt3_true_plot, sigma

    P = _interp_matrix_linear(x_fk, x_plot)
    Cf_plot = P @ Cf_fk @ P.T
    sigma = np.sqrt(np.clip(np.diag(Cf_plot), 0.0, None))

    return x_plot, xt3_true_plot, sigma


def extract_replica_bands(res: dict, mode: str = "global"):
    """
    Extract plotting bands from NN result dictionaries.

    Parameters
    ----------
    res:
        Result dictionary from train_nn_ensemble_forward.

    mode:
        "global":
            Use all trained curves together.
            Uses res["replicas"] with shape (S, Nx).

        "l2_mean":
            First average over NN ensemble members within each L2 replica,
            then compute bands across L2 replicas.
            Uses res["l2_member_replicas"] with shape (Nl2, Nmembers, Nx).

        "nn_within_l2":
            Return per-L2-replica bands over NN members.
            This returns arrays with an extra leading Nl2 axis.

    Returns
    -------
    x, mean, lo68, hi68, lo95, hi95
    """
    x = np.asarray(res["xgrid"], float)

    if mode == "global":
        reps = np.asarray(res["replicas"], float)

        mean = reps.mean(axis=0)
        lo68, hi68 = np.percentile(reps, [16, 84], axis=0)
        lo95, hi95 = np.percentile(reps, [2.5, 97.5], axis=0)

        return x, mean, lo68, hi68, lo95, hi95

    if mode == "l2_mean":
        if "l2_member_replicas" not in res or res["l2_member_replicas"] is None:
            raise KeyError("Result does not contain structured l2_member_replicas.")

        reps = np.asarray(res["l2_member_replicas"], float)
        # shape: (N_l2, N_members, N_x)

        mean_per_l2 = reps.mean(axis=1)
        # shape: (N_l2, N_x)

        mean = mean_per_l2.mean(axis=0)
        lo68, hi68 = np.percentile(mean_per_l2, [16, 84], axis=0)
        lo95, hi95 = np.percentile(mean_per_l2, [2.5, 97.5], axis=0)

        return x, mean, lo68, hi68, lo95, hi95

    if mode == "nn_within_l2":
        if "l2_member_replicas" not in res or res["l2_member_replicas"] is None:
            raise KeyError("Result does not contain structured l2_member_replicas.")

        reps = np.asarray(res["l2_member_replicas"], float)
        # shape: (N_l2, N_members, N_x)

        mean = reps.mean(axis=1)
        lo68, hi68 = np.percentile(reps, [16, 84], axis=1)
        lo95, hi95 = np.percentile(reps, [2.5, 97.5], axis=1)

        return x, mean, lo68, hi68, lo95, hi95

    raise ValueError(f"Unknown mode={mode}")


def gp_uq_bands(
    mu_samples,
    var_f_samples,
    *,
    sigma2_obs_star=0.0,
    draws_per_theta=1,
    seed=0,
):
    rng = np.random.default_rng(seed)

    mu_samples = np.asarray(mu_samples, float)
    var_f_samples = np.asarray(var_f_samples, float)

    if mu_samples.shape != var_f_samples.shape:
        raise ValueError(
            "mu_samples and var_f_samples must have the same shape (S, M)."
        )

    S, M = mu_samples.shape

    mean_curve = mu_samples.mean(axis=0)

    gp_var = var_f_samples.mean(axis=0)
    theta_var = mu_samples.var(axis=0, ddof=1)

    if np.isscalar(sigma2_obs_star):
        noise_var = np.full(M, float(sigma2_obs_star))
    else:
        noise_var = np.asarray(sigma2_obs_star, float)
        if noise_var.shape != (M,):
            raise ValueError("sigma2_obs_star must be scalar or shape (M,)")

    gp_sd = np.sqrt(np.clip(gp_var, 0.0, np.inf))
    theta_sd = np.sqrt(np.clip(theta_var, 0.0, np.inf))

    total_f_var = np.clip(gp_var + theta_var, 0.0, np.inf)
    total_y_var = np.clip(gp_var + theta_var + noise_var, 0.0, np.inf)

    total_f_sd = np.sqrt(total_f_var)
    total_y_sd = np.sqrt(total_y_var)

    z68, z95 = 1.0, 1.96

    bands_gp_mm = (
        mean_curve - z68 * gp_sd,
        mean_curve + z68 * gp_sd,
        mean_curve - z95 * gp_sd,
        mean_curve + z95 * gp_sd,
    )

    bands_f_mm = (
        mean_curve - z68 * total_f_sd,
        mean_curve + z68 * total_f_sd,
        mean_curve - z95 * total_f_sd,
        mean_curve + z95 * total_f_sd,
    )

    bands_y_mm = (
        mean_curve - z68 * total_y_sd,
        mean_curve + z68 * total_y_sd,
        mean_curve - z95 * total_y_sd,
        mean_curve + z95 * total_y_sd,
    )

    K = int(draws_per_theta)
    Mmix = S * K

    f_mix = np.empty((Mmix, M), float)
    y_mix = np.empty((Mmix, M), float)

    noise_sd = np.sqrt(np.clip(noise_var, 0.0, np.inf))

    k = 0
    for s in range(S):
        sd_s = np.sqrt(np.clip(var_f_samples[s], 0.0, np.inf))
        for _ in range(K):
            f = rng.normal(mu_samples[s], sd_s)
            f_mix[k] = f
            y_mix[k] = f + rng.normal(0.0, noise_sd)
            k += 1

    pi_f_68 = np.percentile(f_mix, [16, 84], axis=0)
    pi_f_95 = np.percentile(f_mix, [2.5, 97.5], axis=0)

    pi_y_68 = np.percentile(y_mix, [16, 84], axis=0)
    pi_y_95 = np.percentile(y_mix, [2.5, 97.5], axis=0)

    return {
        "mean_curve": mean_curve,
        "gp_sd": gp_sd,
        "theta_sd": theta_sd,
        "total_f_sd": total_f_sd,
        "total_y_sd": total_y_sd,
        "bands_gp_mm": bands_gp_mm,
        "bands_f_mm": bands_f_mm,
        "bands_y_mm": bands_y_mm,
        "pi_f_68": pi_f_68,
        "pi_f_95": pi_f_95,
        "pi_y_68": pi_y_68,
        "pi_y_95": pi_y_95,
        "gp_var": gp_var,
        "theta_var": theta_var,
        "total_f_var": total_f_var,
        "total_y_var": total_y_var,
    }


def kde_1d(samples, xmin=None, xmax=None, ngrid=400, bw="scott"):
    samples = np.asarray(samples, float)

    if xmin is None:
        xmin = float(samples.min())
    if xmax is None:
        xmax = float(samples.max())

    xs = np.linspace(xmin, xmax, int(ngrid))
    kde = gaussian_kde(samples, bw_method=bw)
    ys = kde(xs)

    return xs, ys


def kde_2d(x, y, xmin=None, xmax=None, ymin=None, ymax=None, ngrid=150, bw="scott"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if xmin is None:
        xmin = float(x.min())
    if xmax is None:
        xmax = float(x.max())
    if ymin is None:
        ymin = float(y.min())
    if ymax is None:
        ymax = float(y.max())

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(ngrid)),
        np.linspace(ymin, ymax, int(ngrid)),
    )

    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    return xx, yy, zz


def _posterior_has(idata, name: str) -> bool:
    return hasattr(idata, "posterior") and (name in idata.posterior.data_vars)


def _get_posterior_1d(idata, name: str):
    if not _posterior_has(idata, name):
        return None
    return np.asarray(idata.posterior[name].values).ravel()


def plot_fig1(
    idata,
    bw1="scott",
    bw2="scott",
    ngrid1=400,
    ngrid2=150,
    figsize=None,
    cb_ranges=None,
    cmap="viridis",
    levels=5,
    alpha_range=(0.0, 5.0),
    beta_range=(0.0, 10.0),
    l0_range=(0.0, 10.0),
    sigma_range=(0.0, 10.0),
):
    alpha = _get_posterior_1d(idata, "alpha")
    beta = _get_posterior_1d(idata, "beta")
    l0 = _get_posterior_1d(idata, "l0")
    sigma = _get_posterior_1d(idata, "sigma")

    if l0 is None or sigma is None:
        present = (
            list(getattr(idata, "posterior", {}).data_vars)
            if hasattr(idata, "posterior")
            else []
        )
        raise KeyError(
            f"plot_fig1 requires 'l0' and 'sigma' in idata.posterior. Present: {present}"
        )

    has_alpha = alpha is not None
    has_beta = beta is not None

    kde_vars = []

    if has_alpha:
        kde_vars.append(("alpha", alpha, alpha_range, r"$\alpha$"))
    if has_beta:
        kde_vars.append(("beta", beta, beta_range, r"$\beta$"))

    kde_vars.append(("l0", l0, l0_range, r"$l$"))
    kde_vars.append(("sigma", sigma, sigma_range, r"$\sigma$"))

    nrows = len(kde_vars)

    if figsize is None:
        figsize = (15, 4 * nrows)

    fig, axes = plt.subplots(nrows, 2, figsize=figsize)

    if nrows == 1:
        axes = np.array([axes])

    def kde1_panel(ax, samples, xmin, xmax, label, xlabel):
        xs, ys = kde_1d(
            samples,
            xmin=float(xmin),
            xmax=float(xmax),
            ngrid=ngrid1,
            bw=bw1,
        )

        ax.plot(xs, ys, lw=2, label=label)
        ax.set_xlim(float(xmin), float(xmax))
        ax.set_xlabel(xlabel)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.5)

    for r, (name, samples, rng, xlabel) in enumerate(kde_vars):
        kde1_panel(
            axes[r, 0],
            samples,
            xmin=rng[0],
            xmax=rng[1],
            label=(
                rf"${{{name}}}$ posterior"
                if name in ("alpha", "beta")
                else rf"{xlabel} posterior"
            ),
            xlabel=xlabel,
        )

    def contour_panel(ax, x, y, key, xmin, xmax, ymin, ymax, xlabel, ylabel):
        xx, yy, zz = kde_2d(
            x,
            y,
            xmin=float(xmin),
            xmax=float(xmax),
            ymin=float(ymin),
            ymax=float(ymax),
            ngrid=ngrid2,
            bw=bw2,
        )

        if cb_ranges and key in cb_ranges:
            vmin, vmax = cb_ranges[key]
        else:
            vmin, vmax = float(zz.min()), float(zz.max())

        cs = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_ticks(np.linspace(vmin, vmax, int(levels)))

        ax.set_xlim(float(xmin), float(xmax))
        ax.set_ylim(float(ymin), float(ymax))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.5)

    right_panels = []

    right_panels.append(
        (
            sigma,
            l0,
            "l0_sigma",
            sigma_range[0],
            sigma_range[1],
            l0_range[0],
            l0_range[1],
            r"$\sigma$",
            r"$l$",
        )
    )

    if has_alpha and has_beta:
        right_panels.append(
            (
                alpha,
                beta,
                "beta_alpha",
                alpha_range[0],
                alpha_range[1],
                beta_range[0],
                beta_range[1],
                r"$\alpha$",
                r"$\beta$",
            )
        )

    if has_alpha:
        right_panels.append(
            (
                alpha,
                l0,
                "l0_alpha",
                alpha_range[0],
                alpha_range[1],
                l0_range[0],
                l0_range[1],
                r"$\alpha$",
                r"$l$",
            )
        )

        right_panels.append(
            (
                alpha,
                sigma,
                "sigma_alpha",
                alpha_range[0],
                alpha_range[1],
                sigma_range[0],
                min(sigma_range[1], 2.0),
                r"$\alpha$",
                r"$\sigma$",
            )
        )

    else:
        right_panels.append(
            (
                l0,
                sigma,
                "sigma_l0",
                l0_range[0],
                l0_range[1],
                sigma_range[0],
                min(sigma_range[1], 2.0),
                r"$l$",
                r"$\sigma$",
            )
        )

    while len(right_panels) < nrows:
        right_panels.append(right_panels[-1])

    right_panels = right_panels[:nrows]

    for r in range(nrows):
        x, y, key, xmin, xmax, ymin, ymax, xlabel, ylabel = right_panels[r]
        contour_panel(
            axes[r, 1],
            x,
            y,
            key,
            xmin,
            xmax,
            ymin,
            ymax,
            xlabel,
            ylabel,
        )

    plt.tight_layout()

    return fig


def plot_fig2(
    x_star,
    mean_curve,
    lo68,
    hi68,
    lo95,
    hi95,
    xt3_true_star=None,
    true_sigma=None,
    outpath=None,
):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    ax = axes[0]

    ax.fill_between(x_star, lo95, hi95, alpha=0.2, color="C0", label="$95\\%$ band")
    ax.fill_between(x_star, lo68, hi68, alpha=0.4, color="C1", label="$68\\%$ band")
    ax.plot(x_star, mean_curve, lw=2, color="C1", label="prediction")

    if xt3_true_star is not None:
        ax.plot(
            x_star,
            xt3_true_star,
            lw=2,
            linestyle="dashed",
            color="C2",
            label="true $xT_3$",
        )

    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
    ax.set_ylabel(r"$xT_3(x)$")
    ax.set_xlim([0, 1.0])
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]

    ax.fill_between(x_star, lo95, hi95, alpha=0.2, color="C0", label="$95\\%$ band")
    ax.fill_between(x_star, lo68, hi68, alpha=0.4, color="C1", label="$68\\%$ band")
    ax.plot(x_star, mean_curve, lw=2, color="C1", label="prediction")

    if xt3_true_star is not None:
        ax.plot(
            x_star,
            xt3_true_star,
            lw=2,
            linestyle="dashed",
            color="C2",
            label="true $xT_3$",
        )

    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
    ax.set_xscale("log")
    ax.set_xlim([1e-5, 1.0])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$xT_3(x)$")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath)

    return fig


def plot_fig2_from_result(
    res: dict,
    mode: str = "global",
    xt3_true_star=None,
    true_sigma=None,
    outpath=None,
):
    """
    Convenience wrapper around plot_fig2.

    mode="global":
      plot total spread over all trained curves.

    mode="l2_mean":
      plot spread across L2 replicas after averaging NN members.
    """
    x, mean, lo68, hi68, lo95, hi95 = extract_replica_bands(res, mode=mode)

    return plot_fig2(
        x,
        mean,
        lo68,
        hi68,
        lo95,
        hi95,
        xt3_true_star=xt3_true_star,
        true_sigma=true_sigma,
        outpath=outpath,
    )


def plot_l2_replica_means(
    res: dict,
    xt3_true_star=None,
    outpath=None,
    alpha=0.25,
):
    """
    Plot the mean curve for each L2 data replica.

    Requires:
      res["l2_member_replicas"] with shape (N_l2, N_members, N_x)
    """
    x = np.asarray(res["xgrid"], float)

    if "l2_member_replicas" not in res or res["l2_member_replicas"] is None:
        raise KeyError("Result does not contain structured l2_member_replicas.")

    reps = np.asarray(res["l2_member_replicas"], float)
    mean_per_l2 = reps.mean(axis=1)

    global_mean = mean_per_l2.mean(axis=0)

    lo68, hi68 = np.percentile(mean_per_l2, [16, 84], axis=0)
    lo95, hi95 = np.percentile(mean_per_l2, [2.5, 97.5], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    for ax in axes:
        ax.fill_between(x, lo95, hi95, alpha=0.2, label="$95\\%$ L2 band")
        ax.fill_between(x, lo68, hi68, alpha=0.4, label="$68\\%$ L2 band")

        for curve in mean_per_l2:
            ax.plot(x, curve, alpha=alpha, lw=1)

        ax.plot(x, global_mean, lw=2, label="mean over L2 replicas")

        if xt3_true_star is not None:
            ax.plot(
                x,
                xt3_true_star,
                lw=2,
                linestyle="dashed",
                label="true $xT_3$",
            )

        ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
        ax.set_ylabel(r"$xT_3(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    axes[0].set_xlim([0, 1.0])

    axes[1].set_xscale("log")
    axes[1].set_xlim([1e-5, 1.0])
    axes[1].set_xlabel(r"$x$")

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath)

    return fig


def plot_single_l2_replica_members(
    res: dict,
    replica_l2: int,
    xt3_true_star=None,
    outpath=None,
    alpha=0.2,
):
    """
    Plot all NN ensemble members belonging to one fixed L2 replica.
    """
    x = np.asarray(res["xgrid"], float)

    if "l2_member_replicas" not in res or res["l2_member_replicas"] is None:
        raise KeyError("Result does not contain structured l2_member_replicas.")

    reps = np.asarray(res["l2_member_replicas"], float)

    if replica_l2 < 0 or replica_l2 >= reps.shape[0]:
        raise ValueError(
            f"replica_l2={replica_l2} out of range. Available: 0 to {reps.shape[0] - 1}"
        )

    curves = reps[replica_l2]
    mean = curves.mean(axis=0)

    lo68, hi68 = np.percentile(curves, [16, 84], axis=0)
    lo95, hi95 = np.percentile(curves, [2.5, 97.5], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    for ax in axes:
        ax.fill_between(x, lo95, hi95, alpha=0.2, label="$95\\%$ NN band")
        ax.fill_between(x, lo68, hi68, alpha=0.4, label="$68\\%$ NN band")

        for curve in curves:
            ax.plot(x, curve, alpha=alpha, lw=1)

        ax.plot(x, mean, lw=2, label=f"mean, L2 replica {replica_l2}")

        if xt3_true_star is not None:
            ax.plot(
                x,
                xt3_true_star,
                lw=2,
                linestyle="dashed",
                label="true $xT_3$",
            )

        ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
        ax.set_ylabel(r"$xT_3(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    axes[0].set_xlim([0, 1.0])

    axes[1].set_xscale("log")
    axes[1].set_xlim([1e-5, 1.0])
    axes[1].set_xlabel(r"$x$")

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath)

    return fig


def plot_fig2_unc(
    x_star,
    mean_curve,
    bands_total,
    bands_ens,
    bands_het,
    xt3_true_star=None,
    true_sigma=None,
    outpath=None,
):
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))

    ax = axes[0]

    lo68, hi68, lo95, hi95 = bands_total

    ax.fill_between(x_star, lo95, hi95, alpha=0.2, label="$95\\%$ total")
    ax.fill_between(x_star, lo68, hi68, alpha=0.4, label="$68\\%$ total")
    ax.plot(x_star, mean_curve, lw=2, label="mean")

    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, label="true $xT_3$")

    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
    ax.set_xlim([0, 1.0])
    ax.set_ylabel(r"$f(x)$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]

    lo68e, hi68e, lo95e, hi95e = bands_ens
    lo68h, hi68h, lo95h, hi95h = bands_het

    ax.fill_between(x_star, lo95h, hi95h, alpha=0.15, label="$95\\%$ het")
    ax.fill_between(x_star, lo68h, hi68h, alpha=0.25, label="$68\\%$ het")
    ax.plot(x_star, mean_curve, lw=2, label="mean")

    if xt3_true_star is not None:
        ax.plot(
            x_star,
            xt3_true_star,
            lw=2,
            linestyle="dashed",
            color="C2",
            label="true $xT_3$",
        )

    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
    ax.set_xscale("log")
    ax.set_xlim([1e-5, 1.0])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[2]

    ax.fill_between(x_star, lo95e, hi95e, alpha=0.15, label="$95\\%$ ensemble")
    ax.fill_between(x_star, lo68e, hi68e, alpha=0.25, label="$68\\%$ ensemble")
    ax.plot(x_star, mean_curve, lw=2, label="mean")

    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, label="NNPDF")

    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
    ax.set_xscale("log")
    ax.set_xlim([1e-5, 1.0])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()

    if outpath:
        fig.savefig(outpath)

    return fig