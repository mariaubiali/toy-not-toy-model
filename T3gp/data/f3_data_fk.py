#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import lhapdf

from validphys.api import API
from validphys.fkparser import load_fktable

THEORYID = 208
PDFSET = "NNPDF40_nnlo_as_01180"
DATASET = "HERA_NC_318GEV_EM-SIGMARED"
DEFAULT_CHANNEL = 9 #9 for proper T3, 4 for T3nn implementation

FULL_BASIS = [
    "photon",  # 0
    "Sigma",   # 1
    "g",       # 2
    "V",       # 3
    "V3",      # 4
    "V8",      # 5
    "V15",     # 6
    "V24",     # 7
    "V35",     # 8
    "T3",      # 9
    "T8",      # 10
    "T15",     # 11
    "T24",     # 12
    "T35",     # 13
]


def infer_process_and_beam(dataset_name):
    ds = dataset_name.upper()

    if ds.startswith("HERACOMB"):
        if "NC" in ds:
            process = "NC"
        elif "CC" in ds:
            process = "CC"
        else:
            raise ValueError(f"Cannot infer process (NC/CC) from dataset: {dataset_name}")

        if "EM" in ds:
            beam = "EM"
        elif "EP" in ds:
            beam = "EP"
        else:
            raise ValueError(f"Cannot infer beam (EM/EP) from dataset: {dataset_name}")

        return process, beam

    if "_NC_" in ds:
        process = "NC"
    elif "_CC_" in ds:
        process = "CC"
    else:
        raise ValueError(f"Cannot infer process (NC/CC) from dataset: {dataset_name}")

    if "_EM-" in ds:
        beam = "EM"
    elif "_EP-" in ds:
        beam = "EP"
    else:
        raise ValueError(f"Cannot infer beam (EM/EP) from dataset: {dataset_name}")

    return process, beam


def load_dataset_table_and_prediction(dataset_name, theoryid, pdfset):
    inp = {
        "dataset_input": {"dataset": dataset_name},
        "use_cuts": "internal",
        "theoryid": theoryid,
        "pdf": pdfset,
    }

    lcd = API.loaded_commondata_with_cuts(**inp)
    pred = np.asarray(API.central_predictions(**inp), dtype=float).reshape(-1)

    table = lcd.commondata_table.reset_index().rename(
        columns={
            "kin1": "x",
            "kin2": "q2",
            "kin3": "y",
            "data": "sigma_data",
        }
    )

    if len(table) != len(pred):
        raise RuntimeError(
            f"Prediction length ({len(pred)}) does not match dataset length ({len(table)})."
        )

    table["sigma_theory_full"] = pred
    table["Yplus"] = 1.0 + (1.0 - table["y"]) ** 2
    table["Yminus"] = 1.0 - (1.0 - table["y"]) ** 2
    return table


def load_fk_table(dataset_name, theoryid):
    ds = API.dataset(
        dataset_input={"dataset": dataset_name},
        use_cuts="internal",
        theoryid=theoryid,
    )
    fk_spec = ds.fkspecs[0]
    print(f"Dataset object: {ds}")
    print(f"Commondata path: {getattr(ds.commondata, 'path', 'unknown')}")
    print(f"FK specs: {ds.fkspecs}")

    return load_fktable(fk_spec)


def load_experimental_covariance(dataset_name, theoryid):
    cov = np.asarray(
        API.dataset_inputs_covmat_from_systematics(
            dataset_inputs=[{"dataset": dataset_name}],
            use_cuts="internal",
            theoryid=theoryid,
        ),
        dtype=float,
    )
    cov = 0.5 * (cov + cov.T)
    return cov


def build_basis_on_xgrid(pdf, xgrid, q):
    out = {name: np.zeros(len(xgrid), dtype=float) for name in FULL_BASIS}

    for i, x in enumerate(xgrid):
        g = pdf.xfxQ(21, x, q)

        u, ub = pdf.xfxQ(2, x, q), pdf.xfxQ(-2, x, q)
        d, db = pdf.xfxQ(1, x, q), pdf.xfxQ(-1, x, q)
        s, sb = pdf.xfxQ(3, x, q), pdf.xfxQ(-3, x, q)
        c, cb = pdf.xfxQ(4, x, q), pdf.xfxQ(-4, x, q)
        b, bb = pdf.xfxQ(5, x, q), pdf.xfxQ(-5, x, q)

        up, dp, sp, cp, bp = u + ub, d + db, s + sb, c + cb, b + bb
        uv, dv, sv, cv, bv = u - ub, d - db, s - sb, c - cb, b - bb

        out["photon"][i] = 0.0
        out["Sigma"][i] = up + dp + sp + cp + bp
        out["g"][i] = g

        out["V"][i] = uv + dv + sv + cv + bv
        out["V3"][i] = uv - dv
        out["V8"][i] = uv + dv - 2.0 * sv
        out["V15"][i] = uv + dv + sv - 3.0 * cv
        out["V24"][i] = uv + dv + sv + cv - 4.0 * bv
        out["V35"][i] = uv + dv + sv + cv + bv

        out["T3"][i] = up - dp
        out["T8"][i] = up + dp - 2.0 * sp
        out["T15"][i] = up + dp + sp - 3.0 * cp
        out["T24"][i] = up + dp + sp + cp - 4.0 * bp
        out["T35"][i] = up + dp + sp + cp + bp

    return out


def contract_channel(fk, pdfset_name, channel):
    sigma = fk.sigma
    if channel not in sigma.columns:
        raise RuntimeError(
            f"Channel {channel} not in FK columns {list(sigma.columns)}"
        )

    basis_name = FULL_BASIS[channel]
    pdf = lhapdf.mkPDF(pdfset_name, 0)
    basis_vals = build_basis_on_xgrid(
        pdf, np.asarray(fk.xgrid, dtype=float), float(fk.Q0)
    )[basis_name]

    s = sigma[channel]
    data_ids = pd.Index(s.index.get_level_values("data")).unique().sort_values()

    out = np.zeros(len(data_ids), dtype=float)
    id_to_pos = {int(id_): i for i, id_ in enumerate(data_ids)}

    for idata in data_ids:
        block = s.xs(idata, level="data")
        x_ids = np.asarray(block.index, dtype=int)
        pos = id_to_pos[int(idata)]
        out[pos] = np.sum(block.to_numpy(dtype=float) * basis_vals[x_ids])

    return basis_name, out


def reconstruct_full_from_fk(fk, pdfset_name):
    pdf = lhapdf.mkPDF(pdfset_name, 0)
    basis = build_basis_on_xgrid(pdf, np.asarray(fk.xgrid, dtype=float), float(fk.Q0))
    sigma = fk.sigma

    data_ids = pd.Index(sigma.index.get_level_values("data")).unique().sort_values()
    out = np.zeros(len(data_ids), dtype=float)

    for channel in sigma.columns:
        s = sigma[channel]
        vals = basis[FULL_BASIS[channel]]
        id_to_pos = {int(id_): i for i, id_ in enumerate(data_ids)}

        for idata in data_ids:
            block = s.xs(idata, level="data")
            x_ids = np.asarray(block.index, dtype=int)
            pos = id_to_pos[int(idata)]
            out[pos] += np.sum(block.to_numpy(dtype=float) * vals[x_ids])

    return out


# def extract_W_matrix(fk, channel):
#     sigma = fk.sigma
#     if channel not in sigma.columns:
#         raise RuntimeError(
#             f"Channel {channel} not in FK columns {list(sigma.columns)}"
#         )

#     W = (
#         sigma[channel]
#         .unstack("x")
#         .sort_index(axis=0)
#         .sort_index(axis=1)
#         .fillna(0.0)
#         .to_numpy(dtype=float)
#     )

#     if not np.isfinite(W).all():
#         bad = np.sum(~np.isfinite(W))
#         raise RuntimeError(f"W matrix for channel {channel} contains {bad} non-finite entries")

#     return W

def extract_W_matrix(fk, channel):
    sigma = fk.sigma
    if channel not in sigma.columns:
        raise RuntimeError(
            f"Channel {channel} not in FK columns {list(sigma.columns)}"
        )

    nx = len(fk.xgrid)

    W_df = (
        sigma[channel]
        .unstack("x")
        .sort_index(axis=0)
        .reindex(columns=np.arange(nx), fill_value=0.0)
        .fillna(0.0)
    )

    W = W_df.to_numpy(dtype=float)

    if not np.isfinite(W).all():
        bad = np.sum(~np.isfinite(W))
        raise RuntimeError(
            f"W matrix for channel {channel} contains {bad} non-finite entries"
        )

    return W


def summarize_array(name, arr):
    arr = np.asarray(arr)
    print(
        f"{name}: shape={arr.shape}, finite={np.isfinite(arr).all()}, "
        f"min={np.nanmin(arr):.6e}, max={np.nanmax(arr):.6e}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theoryid", type=int, default=THEORYID)
    ap.add_argument("--pdfset", type=str, default=PDFSET)
    ap.add_argument("--dataset", type=str, default=DATASET)
    ap.add_argument("--channel", type=int, default=DEFAULT_CHANNEL,
                    help="FK basis channel to isolate, e.g. 9 for T3")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    process, beam = infer_process_and_beam(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Process: {process}")
    print(f"Beam   : {beam}")
    print(f"Channel: {args.channel} ({FULL_BASIS[args.channel]})") 

    df = load_dataset_table_and_prediction(args.dataset, args.theoryid, args.pdfset)
    fk = load_fk_table(args.dataset, args.theoryid)
    c_exp = load_experimental_covariance(args.dataset, args.theoryid)

    xgrid = np.asarray(fk.xgrid, dtype=float)
    q0 = float(fk.Q0)

    print(f"xgrid shape = {xgrid.shape}")
    print(f"x min       = {np.min(xgrid):.6e}")
    print(f"x max       = {np.max(xgrid):.6e}")
    print(f"Q0          = {q0:.6e}")

    print("len(xgrid):", len(fk.xgrid))
    print("max x index in sigma:", fk.sigma.index.get_level_values("x").max())
    sigma_theory_full = df["sigma_theory_full"].to_numpy(dtype=float)
    sigma_reco = reconstruct_full_from_fk(fk, args.pdfset)

    rng = np.random.default_rng(seed=451)
    noise = rng.multivariate_normal(mean=np.zeros(len(sigma_theory_full)),cov=c_exp)
    sigma_em_level1 = sigma_theory_full + noise

    delta = sigma_em_level1 - sigma_theory_full
    chi2 = delta.T @ np.linalg.inv(c_exp) @ delta

    print("L1 sanity chi2 / N =", chi2 / len(delta))

    channel_name, sigma_channel = contract_channel(fk, args.pdfset, args.channel)

    pdf = lhapdf.mkPDF(args.pdfset, 0)
    basis = build_basis_on_xgrid(pdf, xgrid, q0)
    xt3_true = basis[channel_name]

    W_t3 = extract_W_matrix(fk, args.channel)
    y_theory = W_t3 @ xt3_true # y theory for t3
    rng = np.random.default_rng(
    seed=451
    )  # you can set seed if you want reproducible “data”
    noise = rng.multivariate_normal(mean=np.zeros(len(y_theory)), cov=c_exp)

    y_pseudo = y_theory + noise #y-Pseudo for t3

    diff_full = sigma_reco - sigma_theory_full
    diff_channel = y_theory - sigma_channel

    print(f"Selected basis name               : {channel_name}")
    print(f"max |reco - sigma_theory_full|    = {np.max(np.abs(diff_full)):.6e}")
    print(f"max |W @ x_channel - sigma_chan|  = {np.max(np.abs(diff_channel)):.6e}")
    print(f"W_channel shape                   = {W_t3.shape}")
    print(f"c_exp shape                       = {c_exp.shape}")

    summarize_array("x_channel_theory", xt3_true)
    summarize_array("sigma_channel", sigma_channel)
    summarize_array("sigma_channel_from_W", y_theory)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out,
        dataset=np.array([args.dataset]),
        process=np.array([process]),
        beam=np.array([beam]),
        theoryid=np.array([args.theoryid], dtype=int),
        pdfset=np.array([args.pdfset]),
        channel_index=np.array([args.channel], dtype=int),
        channel_name=np.array([channel_name]),

        sigma_data=df["sigma_data"].to_numpy(dtype=float),
        sigma_theory_full=sigma_theory_full,
        sigma_level1=sigma_em_level1,
        sigma_reco=sigma_reco,
        sigma_channel=sigma_channel,
        sigma_channel_from_W=y_pseudo,

        x_vals=df["x"].to_numpy(dtype=float),
        q2_vals=df["q2"].to_numpy(dtype=float),
        y_vals=df["y"].to_numpy(dtype=float),
        Yplus=df["Yplus"].to_numpy(dtype=float),
        Yminus=df["Yminus"].to_numpy(dtype=float),

        q0=np.array([q0], dtype=float),
        xgrid=xgrid,
        x_channel_theory=xt3_true,
        W_channel=W_t3,
        c_exp=c_exp,
    )

    print(f"Wrote {out}")

    # use this for training data generation
    np.savez(
        f"Dataset/hera_data_nc_ep920.npz",
        dataset=np.array([args.dataset]),
        q2_vals=df["q2"].to_numpy(dtype=float),
        y_vals=df["y"].to_numpy(dtype=float),
        y_pseudo=y_pseudo,
        y_theory=y_theory,
        W = W_t3,
        xgrid=xgrid,
        xt3_true=xt3_true,
        c_yy=c_exp,
        Yplus=df["Yplus"].to_numpy(dtype=float),
        Yminus=df["Yminus"].to_numpy(dtype=float),
    )



if __name__ == "__main__":
    main()