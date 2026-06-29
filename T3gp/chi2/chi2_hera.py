#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np


def load_hera(path):
    d = np.load(path)
    files = set(d.files)

    required = {
        "W_channel",
        "sigma_data",
        "sigma_theory_full",
        "xgrid",
        "q0",
        "c_exp",
    }

    missing = sorted(required - files)
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")

    out = {
        "W": np.asarray(d["W_channel"], dtype=float),
        "y_data": np.asarray(d["sigma_data"], dtype=float),
        "y_theory_full": np.asarray(d["sigma_theory_full"], dtype=float),
        "y_L1": np.asarray(d["sigma_level1"], dtype=float),
        "xgrid": np.asarray(d["xgrid"], dtype=float),
        "q0": float(np.asarray(d["q0"]).reshape(-1)[0]),
        "cov": np.asarray(d["c_exp"], dtype=float),
    }

    if "x_channel_theory" in files:
        out["x_true"] = np.asarray(d["x_channel_theory"], dtype=float)

    if "channel_name" in files:
        out["channel_name"] = str(np.asarray(d["channel_name"]).reshape(-1)[0])

    if "dataset" in files:
        out["dataset"] = str(np.asarray(d["dataset"]).reshape(-1)[0])

    return out


def load_model_nn(path):
    d = np.load(path)

    if "xgrid" not in d.files:
        raise KeyError(f"{path} must contain 'xgrid'")
    if "mean_curve" not in d.files:
        raise KeyError(f"{path} must contain 'mean_curve'")

    out = {
        "xgrid": np.asarray(d["xgrid"], dtype=float),
        "mean": np.asarray(d["mean_curve"], dtype=float),
    }

    out["xgrid_cov_ens_f"] = (
        np.asarray(d["xgrid_cov_ens_f"], dtype=float)
        if "xgrid_cov_ens_f" in d.files
        else out["xgrid"]
    )

    out["cov_ens_f"] = (
        np.asarray(d["cov_ens_f"], dtype=float)
        if "cov_ens_f" in d.files and d["cov_ens_f"].size > 0
        else None
    )

    out["var_ens"] = np.asarray(d["var_ens"], dtype=float) if "var_ens" in d.files else None
    out["var_het"] = np.asarray(d["var_het"], dtype=float) if "var_het" in d.files else None

    if out["var_ens"] is not None and out["var_het"] is not None:
        out["std_tot"] = np.sqrt(out["var_ens"] + out["var_het"])
    else:
        out["std_tot"] = None

    out["model_type"] = "nn"

    return out


def load_model_gp(
    path,
    name=None,
    mean_key="mean_curve",
    x_key="x_star",
    replicas_key="replicas",
):
    d = np.load(path)

    required = [x_key, mean_key, replicas_key]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(f"{path} missing required GP keys: {missing}")

    replicas = np.asarray(d[replicas_key], dtype=float)
    mean = np.asarray(d[mean_key], dtype=float)
    xgrid = np.asarray(d[x_key], dtype=float)

    if replicas.ndim < 2:
        raise ValueError(f"{replicas_key} must be at least 2D, got shape {replicas.shape}")

    var_tot = np.var(replicas, axis=0, ddof=1)

    out = {
        "name": name or Path(path).parent.name,
        "xgrid": xgrid,
        "mean": mean,
        "replicas": replicas,
        "var_tot": var_tot,
        "std_tot": np.sqrt(var_tot),
        "model_type": "gp",
    }

    out["xgrid_cov_ens_f"] = (
        np.asarray(d["xgrid_cov_ens_f"], dtype=float)
        if "xgrid_cov_ens_f" in d.files
        else out["xgrid"]
    )

    out["cov_ens_f"] = (
        np.asarray(d["cov_ens_f"], dtype=float)
        if "cov_ens_f" in d.files and d["cov_ens_f"].size > 0
        else None
    )

    return out


def load_model(path):
    try:
        model = load_model_nn(path)
        print(f"[info] Loaded model as NN summary: {path}")
        return model
    except Exception as e_nn:
        print(f"[info] NN loader failed for {path}: {e_nn}")

    try:
        model = load_model_gp(path)
        print(f"[info] Loaded model as GP summary: {path}")
        return model
    except Exception as e_gp:
        print(f"[info] GP loader failed for {path}: {e_gp}")

    raise ValueError(f"Could not load model summary from {path} as either NN or GP format")


def check_strictly_increasing(x, name):
    dx = np.diff(x)
    if not np.all(dx > 0):
        bad = np.where(dx <= 0)[0]
        raise ValueError(f"{name} must be strictly increasing. Bad indices near {bad[:10]}")


def build_interp_matrix(x_src, x_tgt):
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

        if x <= x_src[0]:
            A[i, 0] = 1.0
            continue

        if x >= x_src[-1]:
            A[i, -1] = 1.0
            continue

        j = np.searchsorted(x_src, x) - 1
        j = max(0, min(j, len(x_src) - 2))

        x0, x1 = x_src[j], x_src[j + 1]
        w1 = (x - x0) / (x1 - x0)
        w0 = 1.0 - w1
        A[i, j] = w0
        A[i, j + 1] = w1

    return A


def ensure_symmetric(C):
    return 0.5 * (C + C.T)


def add_jitter(C, rel=1e-12):
    scale = np.max(np.abs(np.diag(C)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return C + (rel * scale) * np.eye(C.shape[0])


def chi2_calculation(C, r):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hera", required=True, help="dataset benchmark npz")
    ap.add_argument("--model", required=True, help="model summary npz")
    ap.add_argument("--out", required=True, help="output npz")
    ap.add_argument("--cov", action="store_true", help="use ensemble covariance")
    args = ap.parse_args()

    hera = load_hera(args.hera)
    model = load_model(args.model)
    ens_cov = args.cov

    xgrid_data = hera["xgrid"]
    W = hera["W"]
    y_data = hera["y_data"]
    y_theory_full = hera["y_theory_full"]
    y_L1 = hera["y_L1"]
    C_exp = hera["cov"]
    x_true = hera.get("x_true", None)

    xgrid_model = model["xgrid"]
    x_model = model["mean"]
    xgrid_cov = model.get("xgrid_cov_ens_f", xgrid_model)

    print("\n--- inputs ---")
    print(f"dataset      = {hera.get('dataset', 'unknown')}")
    print(f"channel_name = {hera.get('channel_name', 'unknown')}")
    print(f"model_type   = {model.get('model_type', 'unknown')}")

    # summarize_array("xgrid_data", xgrid_data)
    # summarize_array("xgrid_model", xgrid_model)
    # summarize_array("x_model", x_model)
    # summarize_matrix("W", W)
    # summarize_matrix("C_exp", C_exp)
    # summarize_array("y_data", y_data)
    # summarize_array("y_theory_full", y_theory_full)
    # summarize_array("y_L1", y_L1)

    ndata = len(y_data)
    nx = len(xgrid_data)

    if W.shape != (ndata, nx):
        raise ValueError(
            f"W shape {W.shape} incompatible with (ndata, nx)=({ndata}, {nx})"
        )
    if C_exp.shape != (ndata, ndata):
        raise ValueError(
            f"C_exp shape {C_exp.shape} incompatible with ({ndata}, {ndata})"
        )

    A = build_interp_matrix(xgrid_model, xgrid_data)
    x_model_data = A @ x_model
    y_t3_model = W @ x_model_data
    # print("W shape: ", W.shape)

    C_model_x = None
    C_model_y = None

    if model.get("cov_ens_f", None) is not None:
        C_model_raw = np.asarray(model["cov_ens_f"], dtype=float)
        A_cov = build_interp_matrix(xgrid_cov, xgrid_data)
        C_model_x = A_cov @ C_model_raw @ A_cov.T
        C_model_x = ensure_symmetric(C_model_x)

        C_model_y = W @ C_model_x @ W.T
        C_model_y = ensure_symmetric(C_model_y)

        # summarize_matrix("C_model_x", C_model_x)
        # summarize_matrix("C_model_y", C_model_y)

    else:
        print("[info] no cov_ens_f found; using experimental covariance only.")

    # summarize_matrix("A", A)
    # summarize_array("x_model_data", x_model_data)
    # summarize_array("y_model_channel", y_t3_model)

    # y_hera_ref = y_theory_full
    y_hera_ref = y_L1

    residual_base = y_data - y_hera_ref

    y_channel_true = None

    if x_true is not None:
        if x_true.shape != (nx,):
            raise ValueError(
                f"x_true shape {x_true.shape} incompatible with nx={nx}"
            )

        # T3 contribution in HERA
        y_t3_hera = W @ x_true

        y_hera_mod = y_hera_ref - y_t3_hera + y_t3_model
        residual = y_data - y_hera_mod

    else:
        print("[warning] no x_true found in benchmark file; using channel-replacement fallback.")
        residual = residual_base.copy()

    # summarize_array("residual_base", residual_base)
    # summarize_array("residual", residual)

    if ens_cov:
        cov = C_exp + C_model_y
        print("use exp + ens as covariance matrix")
    else: 
        cov = C_exp
        print("use only exp as covariance matrix")
    cov = add_jitter(ensure_symmetric(cov))
    chi2 = chi2_calculation(cov, residual)
    chi2_base = chi2_calculation(C_exp, residual_base)

    chi2_per_point = chi2 / ndata
    chi2_base_per_point = chi2_base / ndata
    delta_chi2 = chi2 - chi2_base
    delta_per_point = delta_chi2 / ndata

    delta_n = delta_per_point / (np.sqrt(2/ndata))

    print("\n--- result ---")
    # print(f"chi2 base           = {chi2_base:.12e}")
    # print(f"chi2 model          = {chi2:.12e}")
    # print(f"delta chi2          = {delta_chi2:.12e}")
    print(f"chi2 base / ndata   = {chi2_base_per_point:.12e}")
    print(f"chi2 model / ndata  = {chi2_per_point:.12e}")
    print(f"delta chi2 / ndata  = {delta_per_point}")
    print(f"delta n_sigma       = {delta_n}")
    print(f"ndata               = {ndata}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    save_dict = dict(
        xgrid_data=xgrid_data,
        xgrid_model=xgrid_model,
        xgrid_cov=xgrid_cov,
        A=A,
        W=W,
        C_exp=C_exp,
        x_model=x_model,
        x_model_data=x_model_data,
        y_model_channel=y_t3_model,
        y_data=y_data,
        y_theory_full=y_theory_full,
        residual_base=residual_base,
        residual=residual,
        chi2=np.array([chi2], dtype=float),
        chi2_per_point=np.array([chi2_per_point], dtype=float),
        chi2_base=np.array([chi2_base], dtype=float),
        chi2_base_per_point=np.array([chi2_base_per_point], dtype=float),
        delta_chi2=np.array([delta_chi2], dtype=float),
        delta_per_point=np.array([delta_per_point], dtype=float),
        ndata=np.array([ndata], dtype=int),
    )

    if "dataset" in hera:
        save_dict["dataset"] = np.array([hera["dataset"]])
    if "channel_name" in hera:
        save_dict["channel_name"] = np.array([hera["channel_name"]])

    if x_true is not None:
        save_dict["x_true"] = x_true
    if y_channel_true is not None:
        save_dict["y_channel_true"] = y_channel_true

    np.savez(out, **save_dict)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
