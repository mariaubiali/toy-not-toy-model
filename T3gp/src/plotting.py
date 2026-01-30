from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})

def gp_uq_bands(
    mu_samples,          # (S, M) conditional posterior mean per theta draw
    var_f_samples,       # (S, M) conditional GP variance diag per theta draw
    *,
    sigma2_obs_star=0.0, # scalar or (M,) extra obs noise variance for y* intervals
    draws_per_theta=1,
    seed=0,
):
    """
    Minimal GP uncertainty decomposition + full Bayesian predictive intervals.

    Returns dict with:
      mean_curve
      gp_sd, theta_sd, total_f_sd, total_y_sd
      bands_gp_mm, bands_f_mm, bands_y_mm     # moment-matched (Gaussian) bands
      pi_f_68, pi_f_95, pi_y_68, pi_y_95      # mixture-quantile intervals
    """
    rng = np.random.default_rng(seed)
    mu_samples = np.asarray(mu_samples, float)
    var_f_samples = np.asarray(var_f_samples, float)

    if mu_samples.shape != var_f_samples.shape:
        raise ValueError("mu_samples and var_f_samples must have the same shape (S, M).")

    S, M = mu_samples.shape

    # --- decomposition (law of total variance) ---
    mean_curve = mu_samples.mean(axis=0)
    gp_var = var_f_samples.mean(axis=0)                 # Eθ[Var(f*|θ)]
    theta_var = mu_samples.var(axis=0, ddof=1)          # Varθ[E(f*|θ)]
    total_f_var = gp_var + theta_var
    total_f_sd  = np.sqrt(total_f_var)

    if np.isscalar(sigma2_obs_star):
        noise_var = np.full(M, float(sigma2_obs_star))
    else:
        noise_var = np.asarray(sigma2_obs_star, float)
        if noise_var.shape != (M,):
            raise ValueError("sigma2_obs_star must be scalar or shape (M,)")

    gp_sd = np.sqrt(np.clip(gp_var, 0.0, np.inf))
    theta_sd = np.sqrt(np.clip(theta_var, 0.0, np.inf))
    total_f_sd = np.sqrt(np.clip(gp_var + theta_var, 0.0, np.inf))
    total_y_sd = np.sqrt(np.clip(gp_var + theta_var + noise_var, 0.0, np.inf))
    total_f_var = np.clip(gp_var + theta_var, 0.0, np.inf)
    total_y_var = np.clip(gp_var + theta_var + noise_var, 0.0, np.inf)

    # Moment-matched normal bands (fast)
    z68, z95 = 1.0, 1.96
    bands_gp_mm = (
        mean_curve - z68 * gp_sd, mean_curve + z68 * gp_sd,
        mean_curve - z95 * gp_sd, mean_curve + z95 * gp_sd,
    )
    bands_f_mm = (
        mean_curve - z68 * total_f_sd, mean_curve + z68 * total_f_sd,
        mean_curve - z95 * total_f_sd, mean_curve + z95 * total_f_sd,
    )
    bands_y_mm = (
        mean_curve - z68 * total_y_sd, mean_curve + z68 * total_y_sd,
        mean_curve - z95 * total_y_sd, mean_curve + z95 * total_y_sd,
    )

    # --- full Bayesian predictive intervals (mixture sampling, pointwise) ---
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

        # decomposition components
        "gp_sd": gp_sd,
        "theta_sd": theta_sd,
        "total_f_sd": total_f_sd,
        "total_y_sd": total_y_sd,

        # moment-matched bands
        "bands_gp_mm": bands_gp_mm,
        "bands_f_mm": bands_f_mm,
        "bands_y_mm": bands_y_mm,

        # mixture-quantile intervals (full Bayesian)
        "pi_f_68": pi_f_68,
        "pi_f_95": pi_f_95,
        "pi_y_68": pi_y_68,
        "pi_y_95": pi_y_95,

        #gp + hyperparam unc
        "gp_var": gp_var,
        "theta_var": theta_var,
        "total_f_var": total_f_var,
        "total_y_var": total_y_var,
    }


def kde_1d(samples, xmin=None, xmax=None, ngrid=400, bw="scott"):
    samples = np.asarray(samples, float)
    if xmin is None: xmin = float(samples.min())
    if xmax is None: xmax = float(samples.max())
    xs = np.linspace(xmin, xmax, int(ngrid))
    kde = gaussian_kde(samples, bw_method=bw)
    ys = kde(xs)
    return xs, ys

def kde_2d(x, y, xmin=None, xmax=None, ymin=None, ymax=None, ngrid=150, bw="scott"):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if xmin is None: xmin = float(x.min())
    if xmax is None: xmax = float(x.max())
    if ymin is None: ymin = float(y.min())
    if ymax is None: ymax = float(y.max())

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, int(ngrid)),
        np.linspace(ymin, ymax, int(ngrid)),
    )
    kde = gaussian_kde(np.vstack([x, y]), bw_method=bw)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz

def plot_fig1(
    idata,
    bw1="scott",
    bw2="scott",
    ngrid1=400,
    ngrid2=150,
    figsize=(15, 12),
    cb_ranges=None,
    cmap="viridis",
    levels=5,
):
    # Extract samples (PyMC InferenceData)
    alpha = idata.posterior["alpha"].values.ravel()
    l0    = idata.posterior["l0"].values.ravel()
    sigma = idata.posterior["sigma"].values.ravel()

    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # LEFT column: 1D KDEs
    ax = axes[0, 0]
    xs, ys = kde_1d(alpha, xmin=-0.9, xmax=0.0, ngrid=ngrid1, bw=bw1)
    ax.plot(xs, ys, lw=2, label=r"$\alpha$ posterior")
    ax.set_xlim(-0.9, 0.0)
    # ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(r"$\alpha$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.5)

    ax = axes[1, 0]
    xs, ys = kde_1d(l0, xmin=0.0, xmax=10.0, ngrid=ngrid1, bw=bw1)
    ax.plot(xs, ys, lw=2, label=r"$l$ posterior")
    ax.set_xlim(0.0, 10.0)
    ax.set_xlabel(r"$l$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.5)

    ax = axes[2, 0]
    xs, ys = kde_1d(sigma, xmin=0.0, xmax=10.0, ngrid=ngrid1, bw=bw1)
    ax.plot(xs, ys, lw=2, label=r"$\sigma$ posterior")
    ax.set_xlim(0.0, 10.0)
    # ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(r"$\sigma$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.5)

    # RIGHT column: 2D KDE contour panels
    def contour_panel(ax, x, y, key, xmin, xmax, ymin, ymax, xlabel, ylabel):
        xx, yy, zz = kde_2d(
            x, y,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            ngrid=ngrid2, bw=bw2
        )

        if cb_ranges and key in cb_ranges:
            vmin, vmax = cb_ranges[key]
        else:
            vmin, vmax = float(zz.min()), float(zz.max())

        cs = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)
        cbar = fig.colorbar(cs, ax=ax)

        # match your “set ticks to linear” vibe
        cbar.set_ticks(np.linspace(vmin, vmax, int(levels)))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.5)

    # top-right: l0 vs sigma (your script uses x=sigma, y=l0)
    contour_panel(
        axes[0, 1],
        sigma, l0,
        key="l0_sigma",
        xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0,
        xlabel=r"$\sigma$", ylabel=r"$l$"
    )

    # mid-right: l0 vs alpha (x=alpha, y=l0)
    contour_panel(
        axes[1, 1],
        alpha, l0,
        key="l0_alpha",
        xmin=-0.9, xmax=0.0, ymin=0.0, ymax=10.0,
        xlabel=r"$\alpha$", ylabel=r"$l$"
    )

    # bottom-right: sigma vs alpha (x=alpha, y=sigma)
    contour_panel(
        axes[2, 1],
        alpha, sigma,
        key="sigma_alpha",
        xmin=-0.9, xmax=0.0, ymin=0.0, ymax=2.0,
        xlabel=r"$\alpha$", ylabel=r"$\sigma$"
    )

    plt.tight_layout()
    return fig

def plot_fig2(x_star, mean_curve, lo68, hi68, lo95, hi95, xt3_true_star=None, outpath=None):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    ax = axes[0]
    ax.fill_between(x_star, lo95, hi95, alpha=0.2, color='C0', label="$95\%$ band")
    ax.fill_between(x_star, lo68, hi68, alpha=0.4, color='C1', label="$68\%$ band")
    ax.plot(x_star, mean_curve, lw=2, color='C1', label="prediction")
    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, linestyle='dashed', color='C2', label="NNPDF")
    ax.vlines(x=np.min(x_star), ymin=-0.2, ymax=0.6, color='k', linestyle='dotted', lw=1)
    ax.set_ylabel(r"$xT_3(x)$")
    ax.set_xlim([0, 1.])
    ax.set_ylim([-0.1, 0.5])
    ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(x_star, lo95, hi95, alpha=0.2, color='C0', label="$95\%$ band")
    ax.fill_between(x_star, lo68, hi68, alpha=0.4, color='C1', label="$68\%$ band")
    ax.plot(x_star, mean_curve, lw=2, color='C1', label="prediction")
    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, linestyle='dashed', color='C2', label="NNPDF")
    ax.vlines(x=np.min(x_star), ymin=-0.2, ymax=0.6, color='k', linestyle='dotted', lw=1)
    ax.set_xscale("log")
    ax.set_xlim([1e-3, 1.0])
    ax.set_ylim([-0.1, 0.5])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$xT_3(x)$")
    ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if outpath:
        fig.savefig(outpath)
    return fig

def plot_fig2_unc(
    x_star,
    mean_curve,
    bands_total,   # (lo68, hi68, lo95, hi95)
    bands_ens,     # (lo68, hi68, lo95, hi95)
    bands_het,     # (lo68, hi68, lo95, hi95)
    xt3_true_star=None,
    outpath=None,
):
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))

    # Panel 1: total
    ax = axes[0]
    lo68, hi68, lo95, hi95 = bands_total
    ax.fill_between(x_star, lo95, hi95, alpha=0.2, label="$95\%$ total")
    ax.fill_between(x_star, lo68, hi68, alpha=0.4, label="$68\%$ total")
    ax.plot(x_star, mean_curve, lw=2, label="mean")
    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, label="NNPDF")
    ax.set_xlim([0, 1.0])
    ax.set_ylabel(r"$f(x)$")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    # Panel 2: separate (epistemic vs aleatoric)
    ax = axes[1]
    lo68e, hi68e, lo95e, hi95e = bands_ens
    lo68h, hi68h, lo95h, hi95h = bands_het
    # ax.fill_between(x_star, lo95e, hi95e, alpha=0.15, label="95% ensemble")
    # ax.fill_between(x_star, lo68e, hi68e, alpha=0.25, label="68% ensemble")
    ax.fill_between(x_star, lo95h, hi95h, alpha=0.15, label="$95\%$ het")
    ax.fill_between(x_star, lo68h, hi68h, alpha=0.25, label="$68\%$ het")
    ax.plot(x_star, mean_curve, lw=2, label="mean")
    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, label="NNPDF")
    ax.set_xscale("log")
    ax.set_xlim([1e-5, 1.0])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    # Panel 2: separate (epistemic vs aleatoric)
    ax = axes[2]
    lo68e, hi68e, lo95e, hi95e = bands_ens
    lo68h, hi68h, lo95h, hi95h = bands_het
    ax.fill_between(x_star, lo95e, hi95e, alpha=0.15, label="$95\%$ ensemble")
    ax.fill_between(x_star, lo68e, hi68e, alpha=0.25, label="$68\%$ ensemble")
    # ax.fill_between(x_star, lo95h, hi95h, alpha=0.15, label="95% het")
    # ax.fill_between(x_star, lo68h, hi68h, alpha=0.25, label="68% het")
    ax.plot(x_star, mean_curve, lw=2, label="mean")
    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, label="NNPDF")
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





