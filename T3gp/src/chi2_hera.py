#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_hera(path):
    d = np.load(path)

    out = {
        "xgrid": np.asarray(d["xgrid"], dtype=float),
        "W": np.asarray(d["W_T3"], dtype=float),
        "c_yy": np.asarray(d["c_yy"], dtype=float),
        "y_theory_full": np.asarray(d["y_theory_full"], dtype=float),
        "data": np.asarray(d["data"], dtype=float),
    }

    if "y_pseudo_t3" in d.files:
        out["y_pseudo_t3"] = np.asarray(d["y_pseudo_t3"], dtype=float)

    if "xt3_true" in d.files:
        out["xt3_true"] = np.asarray(d["xt3_true"], dtype=float)

    if "y_theory_t3" in d.files:
        out["y_theory_t3"] = np.asarray(d["y_theory_t3"], dtype=float)

    return out


def load_model(path):
    d = np.load(path)

    if "xgrid" not in d.files:
        raise KeyError(f"{path} must contain 'xgrid'")
    if "mean_curve" not in d.files:
        raise KeyError(f"{path} must contain 'mean_curve'")

    out = {
        "xgrid": np.asarray(d["xgrid"], dtype=float),
        "mean": np.asarray(d["mean_curve"], dtype=float),
    }

    out["var_ens"] = None
    out["var_het"] = None

    if "var_ens" in d.files:
        out["var_ens"] = np.asarray(d["var_ens"], dtype=float)

    if "var_het" in d.files:
        out["var_het"] = np.asarray(d["var_het"], dtype=float)

    if out["var_ens"] is not None and out["var_het"] is not None:
        if out["var_ens"].shape != out["mean"].shape:
            raise ValueError(
                f"var_ens has shape {out['var_ens'].shape}, expected {out['mean'].shape}"
            )
        if out["var_het"].shape != out["mean"].shape:
            raise ValueError(
                f"var_het has shape {out['var_het'].shape}, expected {out['mean'].shape}"
            )

        out["std_tot"] = np.sqrt(out["var_ens"] + out["var_het"])
    else:
        out["std_tot"] = None

    return out


def check_strictly_increasing(x, name):
    dx = np.diff(x)
    if not np.all(dx > 0):
        bad = np.where(dx <= 0)[0]
        raise ValueError(
            f"{name} must be strictly increasing. Bad indices near {bad[:10]}"
        )


def build_interp_matrix(x_src, x_tgt):
    """
    Piecewise-linear interpolation matrix A such that

        y_tgt ~= A @ y_src
    """
    x_src = np.asarray(x_src, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)

    check_strictly_increasing(x_src, "x_src")
    check_strictly_increasing(x_tgt, "x_tgt")

    A = np.zeros((len(x_tgt), len(x_src)), dtype=float)

    for i, x in enumerate(x_tgt):
        exact = np.where(np.isclose(x_src, x, rtol=0.0, atol=0.0))[0]
        if len(exact) > 0:
            A[i, exact[0]] = 1.0
            continue

        if x < x_src[0] or x > x_src[-1]:
            raise ValueError(
                f"Target x={x:.6e} outside source range "
                f"[{x_src[0]:.6e}, {x_src[-1]:.6e}]"
            )

        j = np.searchsorted(x_src, x) - 1
        j = max(0, min(j, len(x_src) - 2))

        x0, x1 = x_src[j], x_src[j + 1]
        dx = x1 - x0
        if dx <= 0 or not np.isfinite(dx):
            raise ValueError(f"Bad source interval: x_src[{j}:{j+2}] = {x0}, {x1}")

        w1 = (x - x0) / dx
        w0 = 1.0 - w1
        A[i, j] = w0
        A[i, j + 1] = w1

    return A

def restrict_to_overlap(xgrid_hera, W_hera, xgrid_nn, xt3_hera=None):
    xmin = max(np.min(xgrid_hera), np.min(xgrid_nn))
    xmax = min(np.max(xgrid_hera), np.max(xgrid_nn))

    if xmin >= xmax:
        raise ValueError(
            f"No overlap between grids: "
            f"xgrid_hera=[{np.min(xgrid_hera):.6e}, {np.max(xgrid_hera):.6e}], "
            f"xgrid_nn=[{np.min(xgrid_nn):.6e}, {np.max(xgrid_nn):.6e}]"
        )

    mask = (xgrid_hera >= xmin) & (xgrid_hera <= xmax)

    if not np.all(mask):
        print(
            f"[info] Restricting xgrid_hera from {len(xgrid_hera)} to {np.sum(mask)} "
            f"points on overlap [{xmin:.6e}, {xmax:.6e}]"
        )

    xgrid_hera_new = xgrid_hera[mask]
    W_hera_new = W_hera[:, mask]
    xt3_hera_new = xt3_hera[mask] if xt3_hera is not None else None

    return xgrid_hera_new, W_hera_new, xt3_hera_new, mask

def ensure_symmetric(C):
    return 0.5 * (C + C.T)


def add_jitter(C, rel=1e-12):
    scale = np.max(np.abs(np.diag(C)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return C + (rel * scale) * np.eye(C.shape[0])


def quad_form_cholesky(C, r):
    L = np.linalg.cholesky(C)
    z = np.linalg.solve(L, r)
    return float(z @ z)


def summarize_array(name, arr):
    arr = np.asarray(arr)
    print(
        f"{name}: shape={arr.shape}, finite={np.isfinite(arr).all()}, "
        f"min={np.nanmin(arr):.6e}, max={np.nanmax(arr):.6e}"
    )


def summarize_matrix(name, M):
    M = np.asarray(M)
    summarize_array(name, M)
    if M.ndim == 2 and M.shape[0] == M.shape[1]:
        sym = np.allclose(M, M.T, rtol=1e-10, atol=1e-12)
        print(f"{name}: symmetric={sym}")
        if np.isfinite(M).all():
            evals = np.linalg.eigvalsh(ensure_symmetric(M))
            print(f"{name}: eig min={evals.min():.6e}, eig max={evals.max():.6e}")


def make_plots(plot_path, xgrid, xt3_hera, xt3_nn, residual, std=None):
    idx = np.arange(len(residual))
    xt3_residual = xt3_hera - xt3_nn

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(plot_path) as pdf:
        fig = plt.figure(figsize=(7, 5))
        plt.plot(xgrid, xt3_hera, label="HERA")
        plt.plot(xgrid, xt3_nn, label="model", color="C1")
        if std is not None:
            lo = xt3_nn - std
            hi = xt3_nn + std
            plt.fill_between(xgrid, lo, hi, alpha=0.3, label="model ±1σ", color="C1")
        plt.xscale("log")
        plt.xlabel("x")
        plt.ylabel("xT3")
        plt.title("xT3 comparison")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 5))
        plt.plot(xgrid, xt3_residual, label="HERA - model")
        plt.axhline(0.0, linewidth=1.0, color="k")
        plt.xscale("log")
        plt.xlabel("x")
        plt.ylabel("Residual")
        plt.title("xT3 residual vs x")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 5))
        plt.plot(idx, residual, label="HERA - model")
        plt.axhline(0.0, linewidth=1.0, color="k")
        plt.xlabel("Data point index")
        plt.ylabel("Residual")
        plt.title("Observable residual")
        plt.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved plots to {plot_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hera", required=True, help="benchmark npz from generator")
    ap.add_argument("--model", required=True, help="model summary npz")
    ap.add_argument("--out", required=True, help="output npz")
    ap.add_argument("--plots", default=None, help="optional PDF output")
    args = ap.parse_args()

    hera = load_hera(args.hera)
    model = load_model(args.model)

    xgrid_hera = hera["xgrid"]
    W_hera = hera["W"]
    cyy_hera = hera["c_yy"]
    y_theory_full = np.asarray(hera["y_theory_full"], dtype=float)
    y_theory_T3 = np.asarray(hera["y_theory_t3"], dtype=float)
    xt3_hera = hera["xt3_true"] if "xt3_true" in hera else None
    exp_data = np.asarray(hera["data"], dtype=float)

    xgrid_nn = model["xgrid"]
    xt3_nn = model["mean"]
    W_nn = np.load('../data/Dataset/W_208.npy')

    print("\n--- inputs ---")
    summarize_array("xgrid_hera", xgrid_hera)
    summarize_array("xgrid_nn", xgrid_nn)
    summarize_array("xt3_nn", xt3_nn)
    summarize_matrix("W_hera", W_hera)
    summarize_matrix("W_nn", W_nn)
    summarize_matrix("cyy_hera", cyy_hera)

    if W_hera.shape != (len(exp_data), len(xgrid_hera)):
        raise ValueError(
            f"W shape {W_hera.shape} incompatible with "
            f"(ndata, nx)=({len(exp_data)}, {len(xgrid_hera)})"
        )
    if cyy_hera.shape != (len(exp_data), len(exp_data)):
        raise ValueError(
            f"cyy_hera shape {cyy_hera.shape} incompatible with data length {len(exp_data)}"
        )
    if len(xgrid_nn) != len(xt3_nn):
        raise ValueError(
            f"model xgrid length {len(xgrid_nn)} incompatible with mean length {len(xt3_nn)}"
        )

    xgrid_hera, W_hera, xt3_hera, mask_hera = restrict_to_overlap(
    xgrid_hera, W_hera, xgrid_nn, xt3_hera
    )

    y_theory_full = y_theory_full.copy()
    y_theory_T3 = y_theory_T3.copy()

    A = build_interp_matrix(xgrid_nn, xgrid_hera)

    print("\n--- projection ---")
    summarize_matrix("A", A)
    summarize_array("A row sums", A.sum(axis=1))

    xt3_nn_hera = A @ xt3_nn
    summarize_array("xt3_nn_hera", xt3_nn_hera)

    std_nn = model.get("std_tot", None)
    if std_nn is not None:
        std_nn_hera = A @ std_nn
    else:
        std_nn_hera = None

    y_t3_nn = W_hera @ xt3_nn_hera
    th_pred = y_theory_full - y_theory_T3 + y_t3_nn

    print("\n--- observables ---")
    summarize_array("model", th_pred)

    residual = exp_data - th_pred
    summarize_array("residual", residual)

    cov = cyy_hera

    cov = add_jitter(cov)
    summarize_matrix("cov after jitter", cov)

    chi2 = quad_form_cholesky(cov, residual)
    ndata = len(exp_data)
    chi2_per_point = chi2 / ndata

    print("\n--- result ---")
    print(f"chi2         = {chi2:.12e}")
    print(f"chi2 / ndata = {chi2_per_point:.12e}")
    print(f"ndata        = {ndata}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out,
        xgrid_hera=xgrid_hera,
        xgrid_nn=xgrid_nn,
        A=A,
        xt3_nn=xt3_nn,
        xt3_nn_hera=xt3_nn_hera,
        exp_data=exp_data,
        th_pred=th_pred,
        residual=residual,
        cyy_hera=cyy_hera,
        cov=cov,
        W_hera=W_hera,
        chi2=np.array([chi2], dtype=float),
        chi2_per_point=np.array([chi2_per_point], dtype=float),
        ndata=np.array([ndata], dtype=int),
    )

    print(f"\nSaved results to {out}")

    if args.plots is not None and xt3_hera is not None:
        if xt3_hera.shape != xt3_nn_hera.shape:
            raise ValueError(
                f"xt3_hera shape {xt3_hera.shape} incompatible with "
                f"xt3_nn_hera shape {xt3_nn_hera.shape}"
            )
        plot_path = out.parent / args.plots
        make_plots(
            plot_path=plot_path,
            xgrid=xgrid_hera,
            xt3_hera=xt3_hera,
            xt3_nn=xt3_nn_hera,
            residual=residual,
            std=std_nn_hera,
        )


if __name__ == "__main__":
    main()