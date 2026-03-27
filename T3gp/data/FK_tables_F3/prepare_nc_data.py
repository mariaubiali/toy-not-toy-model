from copy import copy
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lhapdf
from validphys.api import API
from validphys.coredata import FKTableData
from pineappl.fk_table import FkTable

THEORYID = 208
PDFSET = "NNPDF40_nnlo_as_01180"

DATASET_EM = "HERA_NC_318GEV_EM-SIGMARED"
DATASET_EP = "HERA_NC_318GEV_EP-SIGMARED"
FK_EM = Path(f"data/fktables/{THEORYID}/HERA_NC_318GEV_EM-SIGMARED.pineappl.lz4")
FK_EP = Path(f"data/fktables/{THEORYID}/HERA_NC_318GEV_EP-SIGMARED.pineappl.lz4")

OUT = f"../Dataset/hera_nc_benchmark_{THEORYID}.npz"

# Native full basis order
# [g, T3, T8, T15, Sigma, V3, V8, V15, V]
CODES = [21, 103, 108, 115, 100, 203, 208, 215, 200]
CODE_NAMES = ["g", "T3", "T8", "T15", "Sigma", "V3", "V8", "V15", "V"]
T3_CODE = 103


def local_dis_pineappl_reader(spec, target_xgrid=None):
    """
    Minimal DIS-only replacement for validphys pineappl reader
    adapted to the current pineappl FkTable API.
    Returns FKTableData with xgrid harmonized to target_xgrid if provided.
    """
    pines = [FkTable.read(str(p)) for p in spec.fkpath]
    cfactors = spec.load_cfactors()

    pine_rep = pines[0]
    conv_types = ("UnpolPDF",)

    eko = json.loads(pine_rep.metadata["eko_operator_card"])
    Q0 = float(eko["init"][0])

    if target_xgrid is None:
        xgrid = np.array([])
        for pine in pines:
            xgrid = np.union1d(xgrid, np.asarray(pine.x_grid(), dtype=float))
    else:
        xgrid = np.asarray(target_xgrid, dtype=float)

    xi = np.arange(len(xgrid))
    protected = False

    meta = getattr(spec, "metadata", None)
    shifts = getattr(meta, "shifts", None)
    normalization_per_fktable = getattr(meta, "normalization", None)
    conversion_factor = getattr(meta, "conversion_factor", 1.0)

    def fkname_from_path(p):
        name = Path(p).name
        if name.endswith(".pineappl.lz4"):
            return name[:-len(".pineappl.lz4")]
        if name.endswith(".pineappl"):
            return name[:-len(".pineappl")]
        return Path(p).stem

    fknames = [fkname_from_path(p) for p in spec.fkpath]
    if cfactors is not None:
        cfactors = dict(zip(fknames, cfactors))

    xdivision = xgrid[np.newaxis, np.newaxis, :]
    partial_fktables = []
    ndata = 0

    for fkname, p in zip(fknames, pines):
        cfprod = 1.0
        if cfactors is not None:
            for cfac in cfactors.get(fkname, []):
                cfprod *= cfac.central_value

        raw_fktable = (
            cfprod * np.asarray(p.table()).T / np.asarray(p.bin_normalizations())
        ).T
        n = raw_fktable.shape[0]

        if shifts is not None:
            ndata += shifts.get(fkname, 0)

        if normalization_per_fktable is not None:
            raw_fktable = raw_fktable * normalization_per_fktable.get(fkname, 1.0)

        active_x = np.asarray(p.x_grid(), dtype=float)
        missing_x_points = np.setdiff1d(xgrid, active_x, assume_unique=True)
        for x_point in missing_x_points:
            miss_index = list(xgrid).index(x_point)
            raw_fktable = np.insert(raw_fktable, miss_index, 0.0, axis=2)

        raw_fktable *= conversion_factor / xdivision

        lf = raw_fktable.shape[1]
        if lf != len(CODES):
            raise RuntimeError(
                f"Unexpected number of luminosity channels: got {lf}, expected {len(CODES)}"
            )

        data_idx = np.arange(ndata, ndata + n)
        idx = pd.MultiIndex.from_product([data_idx, xi], names=["data", "x"])

        df_fktable = raw_fktable.swapaxes(0, 1).reshape(lf, -1).T
        partial_fktables.append(pd.DataFrame(df_fktable, columns=CODES, index=idx))

        ndata += n

    sigma = pd.concat(partial_fktables, sort=True, copy=False).fillna(0.0)

    return FKTableData(
        sigma=sigma,
        ndata=ndata,
        Q0=Q0,
        convolution_types=conv_types,
        metadata=meta,
        hadronic=False,
        xgrid=xgrid,
        protected=protected,
    )


def load_fkdata(dataset_name, theoryid, local_fk_path, target_xgrid=None):
    ds = API.dataset(
        dataset_input={"dataset": dataset_name},
        theoryid=theoryid,
        use_cuts="internal",
    )

    spec = copy(ds.fkspecs[0])
    spec.fkpath = [Path(local_fk_path)]
    spec.load_cfactors = lambda: None

    print("Using local FK:", spec.fkpath[0])
    return local_dis_pineappl_reader(spec, target_xgrid=target_xgrid)


def sanitize_covariance(cov, name="cov"):
    cov = np.asarray(cov, dtype=float)
    bad = ~np.isfinite(cov)
    if np.any(bad):
        print(
            f"[warn] {name} contains {int(np.sum(bad))} non-finite entries; replacing with 0."
        )
        cov = cov.copy()
        cov[bad] = 0.0
    cov = 0.5 * (cov + cov.T)
    return cov


def ensure_psd(cov, name="cov", max_tries=10):
    cov = sanitize_covariance(cov, name=name)
    jitter = 1e-12 * max(np.mean(np.diag(cov)), 1.0)

    for i in range(max_tries):
        try:
            np.linalg.cholesky(cov)
            if i > 0:
                print(f"[info] Added diagonal jitter to {name} to make it PSD.")
            return cov
        except np.linalg.LinAlgError:
            cov = cov + np.eye(cov.shape[0]) * jitter
            jitter *= 10

    raise np.linalg.LinAlgError(
        f"Could not make {name} positive definite after {max_tries} attempts."
    )


def native_full_target_2d(pdf, xgrid, q0):
    """
    Build x*f(x,Q0) in the full native basis on xgrid.
    Output shape: (9, nx), ordered as CODES.
    """
    out = np.zeros((len(CODES), len(xgrid)), dtype=float)

    for ix, x in enumerate(xgrid):
        g = pdf.xfxQ(21, x, q0)

        u = pdf.xfxQ(2, x, q0)
        ub = pdf.xfxQ(-2, x, q0)
        d = pdf.xfxQ(1, x, q0)
        db = pdf.xfxQ(-1, x, q0)
        s = pdf.xfxQ(3, x, q0)
        sb = pdf.xfxQ(-3, x, q0)
        c = pdf.xfxQ(4, x, q0)
        cb = pdf.xfxQ(-4, x, q0)

        up = u + ub
        dp = d + db
        sp = s + sb
        cp = c + cb

        uv = u - ub
        dv = d - db
        sv = s - sb
        cv = c - cb

        vals = {
            21: g,
            103: up - dp,
            108: up + dp - 2.0 * sp,
            115: up + dp + sp - 3.0 * cp,
            100: up + dp + sp + cp,
            203: uv - dv,
            208: uv + dv - 2.0 * sv,
            215: uv + dv + sv - 3.0 * cv,
            200: uv + dv + sv + cv,
        }

        for ic, code in enumerate(CODES):
            out[ic, ix] = vals[code]

    return out


def get_channel_index(code):
    if code not in CODES:
        raise ValueError(f"Requested code {code} not found in CODES={CODES}")
    return CODES.index(code)


def main():
    # ------------------------------------------------------------
    # 1. commondata + propagated experimental covariance
    # ------------------------------------------------------------
    inp_em = {
        "dataset_input": {"dataset": DATASET_EM},
        "use_cuts": "internal",
        "theoryid": THEORYID,
    }
    inp_ep = {
        "dataset_input": {"dataset": DATASET_EP},
        "use_cuts": "internal",
        "theoryid": THEORYID,
    }

    lcd_em = API.loaded_commondata_with_cuts(**inp_em)
    lcd_ep = API.loaded_commondata_with_cuts(**inp_ep)

    params_cov = {
        "dataset_inputs": [inp_em["dataset_input"], inp_ep["dataset_input"]],
        "use_cuts": "internal",
        "theoryid": THEORYID,
    }
    cov_full = API.dataset_inputs_covmat_from_systematics(**params_cov)
    cov_full = sanitize_covariance(cov_full, "cov_full")

    df_em = (
        lcd_em.commondata_table.reset_index()
        .rename(
            columns={
                "kin1": "x",
                "kin2": "q2",
                "kin3": "y",
                "data": "sigma_em",
                "entry": "entry_em",
            }
        )
        .assign(idx_em=lambda df: df.index)
    )

    df_ep = (
        lcd_ep.commondata_table.reset_index()
        .rename(
            columns={
                "kin1": "x",
                "kin2": "q2",
                "kin3": "y",
                "data": "sigma_ep",
                "entry": "entry_ep",
            }
        )
        .assign(idx_ep=lambda df: df.index)
    )

    merged_df = (
        df_em.merge(df_ep, on=["x", "q2", "y"], suffixes=("_em", "_ep"))
        .assign(
            Yplus=lambda df: 1.0 + (1.0 - df["y"]) ** 2,
            Yminus=lambda df: 1.0 - (1.0 - df["y"]) ** 2,
        )
    )

    merged_df = merged_df[np.abs(merged_df["Yminus"]) > 1e-12].copy()

    alpha = (merged_df["Yplus"] / (2.0 * merged_df["Yminus"])).to_numpy()

    # Experimental NC xF3-like data vector
    data = alpha * (merged_df["sigma_em"] - merged_df["sigma_ep"])

    q2_vals = merged_df["q2"].to_numpy()
    x_vals = merged_df["x"].to_numpy()
    y_vals = merged_df["y"].to_numpy()
    kinematics = np.column_stack((x_vals, q2_vals, y_vals))

    # covariance propagation to the NC combination
    idx_em_merge = merged_df["idx_em"].to_numpy()
    idx_ep_merge = merged_df["idx_ep"].to_numpy()

    n_em = len(df_em)
    c_emem = cov_full[:n_em, :n_em]
    c_epep = cov_full[n_em:, n_em:]
    c_emep = cov_full[:n_em, n_em:]
    c_epem = cov_full[n_em:, :n_em]

    c_emem_sub = c_emem[np.ix_(idx_em_merge, idx_em_merge)]
    c_epep_sub = c_epep[np.ix_(idx_ep_merge, idx_ep_merge)]
    c_emep_sub = c_emep[np.ix_(idx_em_merge, idx_ep_merge)]
    c_epem_sub = c_epem[np.ix_(idx_ep_merge, idx_em_merge)]

    d_alpha = np.diag(alpha)
    c_yy = d_alpha @ (c_emem_sub + c_epep_sub - c_emep_sub - c_epem_sub) @ d_alpha
    c_yy = ensure_psd(c_yy, name="c_yy")

    # ------------------------------------------------------------
    # 2. exact FK tensor on common HERA xgrid
    # ------------------------------------------------------------
    pine_em = FkTable.read(str(FK_EM))
    pine_ep = FkTable.read(str(FK_EP))
    shared_xgrid = np.union1d(
        np.asarray(pine_em.x_grid(), dtype=float),
        np.asarray(pine_ep.x_grid(), dtype=float),
    )

    fk_em = load_fkdata(DATASET_EM, THEORYID, FK_EM, target_xgrid=shared_xgrid)
    fk_ep = load_fkdata(DATASET_EP, THEORYID, FK_EP, target_xgrid=shared_xgrid)

    W_em = fk_em.get_np_fktable()
    W_ep = fk_ep.get_np_fktable()

    print("W_em raw shape:", W_em.shape)
    print("W_ep raw shape:", W_ep.shape)

    xgrid = np.asarray(fk_em.xgrid, dtype=float)
    Q0 = float(fk_em.Q0)

    if not np.allclose(xgrid, fk_ep.xgrid):
        raise RuntimeError("fk_em.xgrid and fk_ep.xgrid do not match")
    if abs(Q0 - fk_ep.Q0) > 1e-14:
        raise RuntimeError("fk_em.Q0 and fk_ep.Q0 do not match")

    entry_em_rel = merged_df["entry_em"].to_numpy() - 1
    entry_ep_rel = merged_df["entry_ep"].to_numpy() - 1

    # Full native-basis FK tensor for the NC combination: shape (ndata, 9, nx)
    W_tensor = alpha[:, None, None] * (W_em[entry_em_rel] - W_ep[entry_ep_rel])

    idx_t3 = get_channel_index(T3_CODE)
    W_T3 = W_tensor[:, idx_t3, :]

    # ------------------------------------------------------------
    # 3. reference full-basis theory on that xgrid
    # ------------------------------------------------------------
    pdfset = lhapdf.getPDFSet(PDFSET)
    pdf0 = pdfset.mkPDF(0)

    target_full_2d = native_full_target_2d(pdf0, xgrid, Q0)   # (9, nx)
    target_full_flat = target_full_2d.reshape(-1)              # (9*nx,)

    xt3_true = target_full_2d[idx_t3, :]                         # (nx,)

    if W_tensor.shape[2] != len(xgrid):
        raise RuntimeError(
            f"W_tensor last dim is {W_tensor.shape[2]} but len(xgrid)={len(xgrid)}"
        )
    if xt3_true.shape[0] != len(xgrid):
        raise RuntimeError(
            f"T3 target has len {xt3_true.shape[0]} but len(xgrid)={len(xgrid)}"
        )

    # Full HERA theory prediction from all native channels
    y_theory_full = np.einsum("dcx,cx->d", W_tensor, target_full_2d)

    # T3-only contribution to that theory
    y_theory_t3= W_T3 @ xt3_true

    theory_by_channel = np.einsum("dcx,cx->dc", W_tensor, target_full_2d)
    y_theory_full_check = np.sum(theory_by_channel, axis=1)

    reco_max_abs_diff = float(np.max(np.abs(y_theory_full - y_theory_full_check)))
    reco_rel_max_diff = float(
        reco_max_abs_diff / max(np.max(np.abs(y_theory_full_check)), 1e-12)
    )
    rng = np.random.default_rng(
        seed=451
    )
    noise = rng.multivariate_normal(mean=np.zeros(len(y_theory_t3)), cov=c_yy)
    y_pseudo_t3 = y_theory_t3 + noise

    # ------------------------------------------------------------
    # 4. diagnostics
    # ------------------------------------------------------------
    print("\nFinite checks:")
    print("data finite:         ", np.isfinite(data).all())
    print("cov finite:          ", np.isfinite(c_yy).all())
    print("W_tensor finite:     ", np.isfinite(W_tensor).all())
    print("W_T3 finite:         ", np.isfinite(W_T3).all())
    print("target_full_2d finite:", np.isfinite(target_full_2d).all())
    print("xt3_true finite:       ", np.isfinite(xt3_true).all())
    print("theory_full finite:  ", np.isfinite(y_theory_full).all())
    print("y_theory finite:    ", np.isfinite(y_theory_t3).all())

    print("\nShapes:")
    print("data                ", data.shape)
    print("cov                 ", c_yy.shape)
    print("W_tensor            ", W_tensor.shape)
    print("W_T3                ", W_T3.shape)
    print("xgrid               ", xgrid.shape)
    print("Q0                  ", np.array([Q0]).shape)
    print("target_full_flat    ", target_full_flat.shape)
    print("target_full_2d      ", target_full_2d.shape)
    print("xt3_true              ", xt3_true.shape)
    print("y_theory_full         ", y_theory_full.shape)
    print("y_theory           ", y_theory_t3.shape)
    print("theory_by_channel   ", theory_by_channel.shape)
    print("kinematics          ", kinematics.shape)
    print("channel_codes       ", np.asarray(CODES, dtype=int).shape)

    print("\nReconstruction checks:")
    print("max diff:", reco_max_abs_diff)
    print("rel max diff:", reco_rel_max_diff)

    # ------------------------------------------------------------
    # 5. save clean benchmark pack
    # ------------------------------------------------------------
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        OUT,
        data=data,
        c_yy=c_yy,
        y_theory_t3=y_theory_t3,
        y_pseudo_t3=y_pseudo_t3,
        y_theory_full=y_theory_full,
        W_full=W_tensor,
        W_T3=W_T3,
        xgrid=xgrid,
        target_full_2d=target_full_2d,
        xt3_true=xt3_true,
        q2_vals=q2_vals,
        y_vals=y_vals,
        kinematics=kinematics,
        Yplus=merged_df["Yplus"].to_numpy(),
        Yminus=merged_df["Yminus"].to_numpy(),
    )

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()