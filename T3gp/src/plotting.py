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
    P[rows, idx]     = 1.0 - t
    P[rows, idx + 1] = t
    return P


def _truth_cov_fk(W, Cy, lam=None):
    # print("Cy: ", Cy)
    # A = W^T Cy^{-1} W (use Cholesky for speed; assumes Cy SPD)
    L = np.linalg.cholesky(Cy)
    Y = np.linalg.solve(L, W)
    Cy_inv_W = np.linalg.solve(L.T, Y)
    # print("Cy inv: ", Cy_inv_W)
    A = W.T @ Cy_inv_W

    Nx = A.shape[0]
    if lam is None:
        lam = 1e-1 * (np.trace(A) / max(Nx, 1))

    M = A + lam * np.eye(Nx) #what does the lam parameter regularizes?

    # Cov of ridge-GLS estimator: Cf = M^{-1} A M^{-1}
    X = np.linalg.solve(M, A)                 # M^{-1} A
    Cf = X @ np.linalg.solve(M, np.eye(Nx))   # (M^{-1} A) M^{-1}
    return Cf, lam


def select_truth_and_band(ds: dict, x_model: np.ndarray, lam: float | None = None):
    """
    Returns:
      x_plot: grid to plot on (extended if available, else model grid)
      xt3_true_plot: truth on x_plot (or None)
      sigma_true_plot: 1σ truth band on x_plot from (W, Cy) (or None)
    """
    meta = ds.get("meta", {})

    # Base (FK) grid where W is defined
    x_fk = np.asarray(ds["xgrid"], float).ravel()
    W = ds["W"]
    Cy = ds["C"]

    # Choose plot grid + truth
    if "xgrid_ext" in meta and "xt3_ext" in meta:
        x_plot = np.asarray(meta["xgrid_ext"], float).ravel()
        xt3_true_plot = np.asarray(meta["xt3_ext"], float).ravel()
        print("Using extended xgrid")
    else:
        x_plot = np.asarray(x_model, float).ravel()
        xt3_true_plot = np.asarray(meta["xt3_true"], float).ravel() if "xt3_true" in meta else None

    # If no truth, no band
    if xt3_true_plot is None:
        return x_plot, None, None

    # 1) invert on FK grid to get covariance there
    Cf_fk, _lam = _truth_cov_fk(W, Cy, lam=lam)

    # 2) map covariance to plot grid if needed
    if x_plot.shape == x_fk.shape and np.allclose(x_plot, x_fk, rtol=0, atol=0):
        sigma = np.sqrt(np.clip(np.diag(Cf_fk), 0.0, None))
        return x_plot, xt3_true_plot, sigma

    P = _interp_matrix_linear(x_fk, x_plot)
    Cf_plot = P @ Cf_fk @ P.T
    sigma = np.sqrt(np.clip(np.diag(Cf_plot), 0.0, None))
    return x_plot, xt3_true_plot, sigma



def gp_uq_bands(
    mu_samples,  # (S, M) conditional posterior mean per theta draw
    var_f_samples,  # (S, M) conditional GP variance diag per theta draw
    *,
    sigma2_obs_star=0.0,  # scalar or (M,) extra obs noise variance for y* intervals
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
        raise ValueError(
            "mu_samples and var_f_samples must have the same shape (S, M)."
        )

    S, M = mu_samples.shape

    # --- decomposition (law of total variance) ---
    mean_curve = mu_samples.mean(axis=0)
    gp_var = var_f_samples.mean(axis=0)  # Eθ[Var(f*|θ)]
    theta_var = mu_samples.var(axis=0, ddof=1)  # Varθ[E(f*|θ)]
    total_f_var = gp_var + theta_var
    total_f_sd = np.sqrt(total_f_var)

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
        # gp + hyperparam unc
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
    """Return flattened posterior samples for `name`, or None if absent."""
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
    # ---- Extract samples safely ----
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

    # ---- Decide how many 1D KDE rows we have ----
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
        axes = np.array([axes])  # ensure 2D indexing

    # ---- LEFT column: 1D KDEs ----
    def kde1_panel(ax, samples, xmin, xmax, label, xlabel):
        xs, ys = kde_1d(
            samples, xmin=float(xmin), xmax=float(xmax), ngrid=ngrid1, bw=bw1
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

    # ---- RIGHT column: 2D KDE contour panels ----
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

        # cb_ranges (optional)
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

    # Choose which pair-plots to show, aligned with rows.
    # We always have l0 and sigma; alpha/beta are optional.
    # Strategy:
    # - First row: sigma vs l0 (always)
    # - If beta exists: alpha vs beta
    # - If alpha exists: alpha vs l0, alpha vs sigma
    # - Otherwise: repeat informative pairs (sigma vs l0, or l0 vs sigma) to fill rows

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
        # No alpha: fall back to l0 vs sigma in different orientations
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

    # Ensure we have exactly nrows panels on the right (repeat last if needed)
    while len(right_panels) < nrows:
        right_panels.append(right_panels[-1])
    right_panels = right_panels[:nrows]

    for r in range(nrows):
        x, y, key, xmin, xmax, ymin, ymax, xlabel, ylabel = right_panels[r]
        contour_panel(axes[r, 1], x, y, key, xmin, xmax, ymin, ymax, xlabel, ylabel)

    plt.tight_layout()
    return fig


def plot_fig2(x_star, mean_curve, lo68, hi68, lo95, hi95, xt3_true_star=None,true_sigma=None, outpath=None):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    ax = axes[0]
    ax.fill_between(x_star, lo95, hi95, alpha=0.2, color='C0', label="$95\%$ band")
    ax.fill_between(x_star, lo68, hi68, alpha=0.4, color='C1', label="$68\%$ band")
    ax.plot(x_star, mean_curve, lw=2, color='C1', label="prediction")
    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, linestyle='dashed', color='C2', label="true $xT_3$")

        # if true_sigma is not None:
        #     ax.fill_between(
        #         x_star,
        #         xt3_true_star - 1.96 * true_sigma,
        #         xt3_true_star + 1.96 * true_sigma,
        #         alpha=0.12, color="C2", label="true $95\\%$ (from $C_Y$)"
        #     )
        #     ax.fill_between(
        #         x_star,
        #         xt3_true_star - 1.00 * true_sigma,
        #         xt3_true_star + 1.00 * true_sigma,
        #         alpha=0.20, color="C2", label="true $68\\%$ (from $C_Y$)"
        #     )
    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
    ax.set_ylabel(r"$xT_3(x)$")
    ax.set_xlim([0, 1.])
    # ax.set_ylim([-0.1, 0.5])
    ax.legend(frameon=False, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(x_star, lo95, hi95, alpha=0.2, color='C0', label="$95\%$ band")
    ax.fill_between(x_star, lo68, hi68, alpha=0.4, color='C1', label="$68\%$ band")
    ax.plot(x_star, mean_curve, lw=2, color='C1', label="prediction")
    if xt3_true_star is not None:
        ax.plot(x_star, xt3_true_star, lw=2, linestyle='dashed', color='C2', label="true $xT_3$")

        # if true_sigma is not None:
        #     ax.fill_between(
        #         x_star,
        #         xt3_true_star - 1.96 * true_sigma,
        #         xt3_true_star + 1.96 * true_sigma,
        #         alpha=0.12, color="C2", label="true $95\\%$ (from $C_Y$)"
        #     )
        #     ax.fill_between(
        #         x_star,
        #         xt3_true_star - 1.00 * true_sigma,
        #         xt3_true_star + 1.00 * true_sigma,
        #         alpha=0.20, color="C2", label="true $68\\%$ (from $C_Y$)"
        #     )
    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
    ax.set_xscale("log")
    ax.set_xlim([1e-5, 1.0])
    # ax.set_ylim([-0.1, 0.5])
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
    true_sigma=None,
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
        ax.plot(x_star, xt3_true_star, lw=2, label="true $xT_3$")
    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
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
        ax.plot(x_star, xt3_true_star, lw=2, linestyle='dashed', color='C2', label="true $xT_3$")

        # if true_sigma is not None:
        #     ax.fill_between(
        #         x_star,
        #         xt3_true_star - 1.96 * true_sigma,
        #         xt3_true_star + 1.96 * true_sigma,
        #         alpha=0.12, color="C2", label="true $95\\%$ (from $C_Y$)"
        #     )
        #     ax.fill_between(
        #         x_star,
        #         xt3_true_star - 1.00 * true_sigma,
        #         xt3_true_star + 1.00 * true_sigma,
        #         alpha=0.20, color="C2", label="true $68\\%$ (from $C_Y$)"
        #     )
    ax.vlines(x=0.069, ymin=-0.2, ymax=0.6, color="k", linestyle="dotted", lw=1)
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

        # if true_sigma is not None:
        #     ax.fill_between(
        #         x_star,
        #         xt3_true_star - 1.96 * true_sigma,
        #         xt3_true_star + 1.96 * true_sigma,
        #         alpha=0.12, color="C2", label="true $95\\%$ (from $C_Y$)"
        #     )
        #     ax.fill_between(
        #         x_star,
        #         xt3_true_star - 1.00 * true_sigma,
        #         xt3_true_star + 1.00 * true_sigma,
        #         alpha=0.20, color="C2", label="true $68\\%$ (from $C_Y$)"
        #     )
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