from copy import copy
from pathlib import Path
import pandas as pd
import json

from validphys.api import API
from validphys.coredata import FKTableData
from pineappl.fk_table import FkTable
import numpy as np
import lhapdf


THEORYID = 208
PDFSET = "NNPDF40_nnlo_as_01180"

DATASET_EM = "HERA_NC_318GEV_EM-SIGMARED"
DATASET_EP = "HERA_NC_318GEV_EP-SIGMARED"
FK_EM = Path(f"data/fktables/{THEORYID}/HERA_NC_318GEV_EM-SIGMARED.pineappl.lz4")
FK_EP = Path(f"data/fktables/{THEORYID}/HERA_NC_318GEV_EP-SIGMARED.pineappl.lz4")

OUT = f"../Dataset/data_NC_{THEORYID}.npz"

# Native full basis order
# [g, T3, T8, T15, Sigma, V3, V8, V15, V]
CODES = [21, 103, 108, 115, 100, 203, 208, 215, 200]
CODE_NAMES = ["g", "T3", "T8", "T15", "Sigma", "V3", "V8", "V15", "V"]

# ------------------------------------------------------------
# Toy-model coefficients for effective xF3 combination.
# Recommended simple F3-like choice:
#   uv + dv
# ------------------------------------------------------------
coeff_u = 1.0
coeff_d = 1.0
coeff_s = 0.0
coeff_c = 0.0


def local_dis_pineappl_reader(spec, target_xgrid=None):
    """
    Minimal DIS-only replacement for validphys pineappl reader
    adapted to the current pineappl FkTable API.
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


def native_full_target(pdf, xgrid, q0):
    """
    Build target in full native basis:
    [g, T3, T8, T15, Sigma, V3, V8, V15, V]
    returned flattened channel-major, i.e. shape (9 * nx,).
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

    return out.reshape(-1)


def reshape_target(flat_target, xgrid, ncodes=len(CODES)):
    """
    Inverse of channel-major flattening.
    flat_target shape: (ncodes * nx,)
    returns shape: (ncodes, nx)
    """
    nx = len(xgrid)
    if flat_target.shape[0] != ncodes * nx:
        raise ValueError(
            f"Cannot reshape target of size {flat_target.shape[0]} into ({ncodes}, {nx})."
        )
    return flat_target.reshape(ncodes, nx)


def basis_to_valence_components(target_2d):
    """
    Convert [V3, V8, V15, V] basis pieces into quark valence components.
    target_2d shape: (9, nx)
    returns uv, dv, sv, cv each shape: (nx,)
    """
    V3 = target_2d[CODES.index(203)]
    V8 = target_2d[CODES.index(208)]
    V15 = target_2d[CODES.index(215)]
    V = target_2d[CODES.index(200)]

    uv = 0.25 * V + (1.0 / 12.0) * V15 + 0.5 * V3 + (1.0 / 6.0) * V8
    dv = 0.25 * V + (1.0 / 12.0) * V15 - 0.5 * V3 + (1.0 / 6.0) * V8
    sv = 0.25 * V + (1.0 / 12.0) * V15 - (1.0 / 3.0) * V8
    cv = 0.25 * V - 0.25 * V15

    return uv, dv, sv, cv


def basis_operator_to_valence_components(W_tensor):
    """
    Convert [V3, V8, V15, V] operator pieces into quark valence operators.
    W_tensor shape: (ndata, 9, nx)
    returns W_uv, W_dv, W_sv, W_cv each shape: (ndata, nx)
    """
    W_V3 = W_tensor[:, CODES.index(203), :]
    W_V8 = W_tensor[:, CODES.index(208), :]
    W_V15 = W_tensor[:, CODES.index(215), :]
    W_V = W_tensor[:, CODES.index(200), :]

    W_uv = 0.25 * W_V + (1.0 / 12.0) * W_V15 + 0.5 * W_V3 + (1.0 / 6.0) * W_V8
    W_dv = 0.25 * W_V + (1.0 / 12.0) * W_V15 - 0.5 * W_V3 + (1.0 / 6.0) * W_V8
    W_sv = 0.25 * W_V + (1.0 / 12.0) * W_V15 - (1.0 / 3.0) * W_V8
    W_cv = 0.25 * W_V - 0.25 * W_V15

    return W_uv, W_dv, W_sv, W_cv


def combine_valence_target(target_2d):
    """
    Build a single effective xF3-like target from quark valence components.
    """
    uv, dv, sv, cv = basis_to_valence_components(target_2d)
    return coeff_u * uv + coeff_d * dv + coeff_s * sv + coeff_c * cv


def combine_valence_operator(W_tensor):
    """
    Build a single effective xF3-like operator from quark valence components.
    """
    W_uv, W_dv, W_sv, W_cv = basis_operator_to_valence_components(W_tensor)
    return coeff_u * W_uv + coeff_d * W_dv + coeff_s * W_sv + coeff_c * W_cv


def main():
    # ------------------------------------------------------------
    # 1. commondata + covariance from API
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

    # NC xF3-like observable
    y_vals = alpha * (merged_df["sigma_em"] - merged_df["sigma_ep"])

    q2_vals = merged_df["q2"].to_numpy()
    x_vals = merged_df["x"].to_numpy()
    y_kin_vals = merged_df["y"].to_numpy()
    kinematics = np.column_stack((x_vals, q2_vals, y_kin_vals))

    # covariance propagation
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
    # 2. exact FK data via pineappl_reader
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

    # full tensor operator: (ndata_matched, 9, nx)
    W_tensor = alpha[:, None, None] * (W_em[entry_em_rel] - W_ep[entry_ep_rel])

    # reduced effective operator: (ndata_matched, nx)
    W = combine_valence_operator(W_tensor)

    # ------------------------------------------------------------
    # 3. exact native target on xgrid
    # ------------------------------------------------------------
    pdfset = lhapdf.getPDFSet(PDFSET)
    pdf0 = pdfset.mkPDF(0)

    xt3_true_full = native_full_target(pdf0, xgrid, Q0)
    xt3_true_full_2d = reshape_target(xt3_true_full, xgrid, ncodes=len(CODES))

    uv_true, dv_true, sv_true, cv_true = basis_to_valence_components(xt3_true_full_2d)

    # reduced effective target: (nx,)
    xt3_true = combine_valence_target(xt3_true_full_2d)

    if xt3_true.shape[0] != W.shape[1]:
        raise RuntimeError(
            f"Shape mismatch: xt3_true has length {xt3_true.shape[0]}, but W expects {W.shape[1]}."
        )

    y_theory = W @ xt3_true

    # direct contraction cross-check with reduced representation
    y_direct = np.einsum("dx,x->d", W, xt3_true)

    reco_max_abs_diff = float(np.max(np.abs(y_theory - y_direct)))
    reco_rel_max_diff = float(
        reco_max_abs_diff / max(np.max(np.abs(y_direct)), 1e-12)
    )

    # pseudo-data
    rng = np.random.default_rng(seed=451)
    noise = rng.multivariate_normal(mean=np.zeros(len(y_theory)), cov=c_yy)
    y_pseudo = y_theory + noise

    # ------------------------------------------------------------
    # 4. diagnostics
    # ------------------------------------------------------------
    print("\nEffective xF3 coefficients in valence basis:")
    print("coeff_u =", coeff_u)
    print("coeff_d =", coeff_d)
    print("coeff_s =", coeff_s)
    print("coeff_c =", coeff_c)

    print("\nFinite checks:")
    print("W finite:", np.isfinite(W).all())
    print("xt3_true finite:", np.isfinite(xt3_true).all())
    print("y_theory finite:", np.isfinite(y_theory).all())
    print("y_direct finite:", np.isfinite(y_direct).all())
    print("c_yy finite:", np.isfinite(c_yy).all())

    print("\nShapes:")
    print("q2_vals             ", q2_vals.shape)
    print("x_vals              ", x_vals.shape)
    print("y_kin_vals          ", y_kin_vals.shape)
    print("y_vals              ", y_vals.shape)
    print("y_pseudo            ", y_pseudo.shape)
    print("W                   ", W.shape)
    print("W_tensor            ", W_tensor.shape)
    print("xgrid               ", xgrid.shape)
    print("Q0                  ", np.array([Q0]).shape)
    print("y_theory            ", y_theory.shape)
    print("y_direct            ", y_direct.shape)
    print("xt3_true            ", xt3_true.shape)
    print("xt3_true_full       ", xt3_true_full.shape)
    print("xt3_true_full_2d    ", xt3_true_full_2d.shape)
    print("uv_true             ", uv_true.shape)
    print("dv_true             ", dv_true.shape)
    print("sv_true             ", sv_true.shape)
    print("cv_true             ", cv_true.shape)
    print("c_yy                ", c_yy.shape)
    print("kinematics          ", kinematics.shape)
    print("Yplus               ", merged_df['Yplus'].to_numpy().shape)
    print("Yminus              ", merged_df['Yminus'].to_numpy().shape)
    print("channel_codes       ", np.asarray(CODES, dtype=int).shape)

    print("\nReconstruction checks:")
    print("max diff:", reco_max_abs_diff)
    print("rel max diff:", reco_rel_max_diff)

    # ------------------------------------------------------------
    # 5. save
    # ------------------------------------------------------------
    Path("Dataset").mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        # reduced effective core objects
        q2_vals=q2_vals,
        x_vals=x_vals,
        y_vals=y_vals,
        y_pseudo=y_pseudo,
        W=W,
        xgrid=xgrid,
        y_theory=y_theory,
        xt3_true=xt3_true,
        c_yy=c_yy,
        kinematics=kinematics,

        # extras
        y_kin_vals=y_kin_vals,
        Q0=np.array([Q0]),
        y_direct=y_direct,
        Yplus=merged_df["Yplus"].to_numpy(),
        Yminus=merged_df["Yminus"].to_numpy(),
        channel_codes=np.asarray(CODES, dtype=int),
        channel_names=np.asarray(CODE_NAMES, dtype="U16"),
        xt3_true_full=xt3_true_full,
        xt3_true_full_2d=xt3_true_full_2d,
        W_tensor=W_tensor,
        uv_true=uv_true,
        dv_true=dv_true,
        sv_true=sv_true,
        cv_true=cv_true,
        coeff_u=np.array([coeff_u], dtype=float),
        coeff_d=np.array([coeff_d], dtype=float),
        coeff_s=np.array([coeff_s], dtype=float),
        coeff_c=np.array([coeff_c], dtype=float),
        reco_max_abs_diff=np.array([reco_max_abs_diff]),
        reco_rel_max_diff=np.array([reco_rel_max_diff]),
    )

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()