import lhapdf

import numpy as np
import argparse

from validphys.api import API
from validphys.pineparser import pineappl_reader
from validphys.fkparser import load_fktable
from validphys.loader import Loader
from validphys import covmats
from types import SimpleNamespace
import matplotlib.pyplot as plt

def project_pd_cov_to_difference(cov_full, idx_p_merge, idx_d_merge, n_p):
    """
    Project covariance of concatenated [F2_p, F2_d] data to covariance of

        y = F2_p - F2_d

    for the matched proton/deuteron rows.
    """
    cov_full = np.asarray(cov_full, dtype=float)

    c_pp = cov_full[:n_p, :n_p]
    c_dd = cov_full[n_p:, n_p:]
    c_pd = cov_full[:n_p, n_p:]
    c_dp = cov_full[n_p:, :n_p]

    c_pp_sub = c_pp[np.ix_(idx_p_merge, idx_p_merge)]
    c_dd_sub = c_dd[np.ix_(idx_d_merge, idx_d_merge)]
    c_pd_sub = c_pd[np.ix_(idx_p_merge, idx_d_merge)]
    c_dp_sub = c_dp[np.ix_(idx_d_merge, idx_p_merge)]

    c_yy = c_pp_sub + c_dd_sub - c_pd_sub - c_dp_sub
    return 0.5 * (c_yy + c_yy.T)


def make_t0_predictions_from_ypseudo(
    lcd_p,
    lcd_d,
    idx_p_merge,
    idx_d_merge,
    y_vals,
    y_theory,
):
    """
    Build validphys-compatible T0 predictions for the original proton and
    deuteron datasets, using the reduced pseudo-data as the T0 reference for

        y = F2_p - F2_d.

    The symmetric prescription shifts both F2_p and F2_d by half of the
    realised pseudo-data fluctuation.
    """
    t0_pred_p = lcd_p.commondata_table["data"].to_numpy(dtype=float).copy()
    t0_pred_d = lcd_d.commondata_table["data"].to_numpy(dtype=float).copy()

    delta_y = y_theory - y_vals

    t0_pred_p[idx_p_merge] += 0.5 * delta_y
    t0_pred_d[idx_d_merge] -= 0.5 * delta_y

    assert np.allclose(
        t0_pred_p[idx_p_merge] - t0_pred_d[idx_d_merge],
        y_theory,
    )

    return [t0_pred_p, t0_pred_d]


def build_c_t0_yy_from_validphys(
    lcd_p,
    lcd_d,
    idx_p_merge,
    idx_d_merge,
    n_p,
    y_vals,
    y_theory,
):
    """
    Colibri-style validphys T0 covariance construction, projected to the
    reduced T3 observable y = F2_p - F2_d.
    """
    dataset_inputs_t0_predictions = make_t0_predictions_from_ypseudo(
        lcd_p=lcd_p,
        lcd_d=lcd_d,
        idx_p_merge=idx_p_merge,
        idx_d_merge=idx_d_merge,
        y_vals=y_vals,
        y_theory=y_theory,
    )

    # Low-level validphys expects DatasetInput-like objects with .weight
    # Colibri passes data.dsinputs. Here we only need unit weights.
    dsinp_p = SimpleNamespace(weight=1.0)
    dsinp_d = SimpleNamespace(weight=1.0)

    cov_t0_full = covmats.dataset_inputs_t0_covmat_from_systematics(
        [lcd_p, lcd_d],
        data_input=[dsinp_p, dsinp_d],
        use_weights_in_covmat=False,
        norm_threshold=None,
        dataset_inputs_t0_predictions=dataset_inputs_t0_predictions,
    )

    cov_t0_full = np.asarray(cov_t0_full, dtype=float)

    c_t0_yy = project_pd_cov_to_difference(
        cov_full=cov_t0_full,
        idx_p_merge=idx_p_merge,
        idx_d_merge=idx_d_merge,
        n_p=n_p,
    )

    return c_t0_yy, cov_t0_full, dataset_inputs_t0_predictions

loader = Loader()

parser = argparse.ArgumentParser(
    description="Run BCDMS analysis with a fixed theory ID"
)

# parser.add_argument(
#     "--theoryid",
#     type=int,
#     default=208,
#     choices=[200, 208, 40001000],
#     help="Theory ID (allowed: 208 or 40001000)",
# )

# args = parser.parse_args()
theoryid = 208

t3_index = 2  # flavor index in FK table
mp = 0.938
mp2 = mp**2

# Generate data based on T3nn
# define input datasets
inp_p = {
    "dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2", "variant": "legacy"},
    "use_cuts": "internal",
    "theoryid": theoryid,
}
inp_d = {
    "dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_D_EM-F2", "variant": "legacy"},
    "use_cuts": "internal",
    "theoryid": theoryid,
}

lcd_p = API.loaded_commondata_with_cuts(**inp_p)
lcd_d = API.loaded_commondata_with_cuts(**inp_d)


pdfset = lhapdf.getPDFSet(
    "NNPDF40_nnlo_as_01180"
)  # This PDF can be changed to any toy underlying PDF set

if theoryid in {200, 208}:
    fk_p = load_fktable(
        loader.check_fktable(setname="BCDMSP", theoryID=theoryid, cfac=())
    )
    fk_d = load_fktable(
        loader.check_fktable(setname="BCDMSD", theoryID=theoryid, cfac=())
    )

elif theoryid == 40001000:
    ds_p = API.dataset(
        dataset_input={"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2"},
        theoryid=theoryid,
        use_cuts="internal",
    )
    ds_d = API.dataset(
        dataset_input={"dataset": "BCDMS_NC_NOTFIXED_D_EM-F2"},
        theoryid=theoryid,
        use_cuts="internal",
    )

    # Read PineAPPL FK tables for each dataset (often multiple sqrt(s) grids)
    fk_p_list = [pineappl_reader(fkspec) for fkspec in ds_p.fkspecs]
    fk_d_list = [pineappl_reader(fkspec) for fkspec in ds_d.fkspecs]

    fk_p = fk_p_list[0]
    fk_d = fk_d_list[0]


params_cov = {
    "dataset_inputs": [inp_p["dataset_input"], inp_d["dataset_input"]],
    "use_cuts": "internal",
    "theoryid": theoryid,
}
cov_full = API.dataset_inputs_covmat_from_systematics(**params_cov)

BCDMS_BEAM_ENERGIES = np.array([100, 120, 200, 280])

def snap_beam_energy(e_raw):
    """Map computed beam energies to nominal BCDMS values within 5%."""
    print("in snap beam energy")
    result = np.empty_like(e_raw)
    for i, e in enumerate(e_raw):
        rel_diff = np.abs(BCDMS_BEAM_ENERGIES - e) / BCDMS_BEAM_ENERGIES
        best = np.argmin(rel_diff)
        if rel_diff[best] > 0.05:
            import warnings
            warnings.warn(f"Computed E_beam={e:.1f} not within 5% of any nominal energy")
            result[i] = round(e)
        else:
            result[i] = BCDMS_BEAM_ENERGIES[best]
    return result

df_p = (
    lcd_p.commondata_table.reset_index()
    .rename(
        columns={
            "kin1": "x",
            "kin2": "q2",
            "kin3": "y",
            "data": "F2_p",
            "stat": "error",
            "entry": "entry_p",
        },
    )
    .assign(
        idx_p=lambda df: df.index,
        E_beam=lambda df: snap_beam_energy(
            (df["q2"] / (2 * mp * df["x"] * df["y"])).to_numpy()
        ),
    )
)
df_d = (
    lcd_d.commondata_table.reset_index()
    .rename(
        columns={
            "kin1": "x",
            "kin2": "q2",
            "kin3": "y",
            "data": "F2_d",
            "stat": "error",
            "entry": "entry_d",
        },
    )
    .assign(
        idx_d=lambda df: df.index,
        E_beam=lambda df: snap_beam_energy(
            (df["q2"] / (2 * mp * df["x"] * df["y"])).to_numpy()
        ),
    )
)


# Merge datasets
# modify merge function to remove double counting
merged_df = df_p.merge(df_d, on=["x", "q2", "E_beam"], suffixes=("_p", "_d")).assign(
    y_val=lambda df: (df["F2_p"] - df["F2_d"]),
    F2_d=lambda df: (df["F2_d"]),
    w2=lambda df: df["q2"] * (1 - df["x"]) / df["x"] + mp2,
)

# remove duplicates to match 248 entries as in paper, for full 611 entries just comment out this line
# merged_df = merged_df.groupby(["x", "q2", "F2_d", "entry_d"]).first().reset_index()

# Extract q2_vals and y_real for later use
q2_vals = merged_df["q2"].to_numpy()
y_vals = merged_df["y_val"].to_numpy()
x_data = merged_df["x"].to_numpy()

# print(x_data)
# print(q2_vals)
# print(merged_df["F2_d"].to_numpy())

# print(np.unique(q2_vals))

kinematics = np.column_stack((merged_df["x"].to_numpy(), merged_df["q2"].to_numpy()))

# Calculate FK tables in here
wp = fk_p.get_np_fktable()  # shape (n_data_fk, n_flav, n_grid)
wd = fk_d.get_np_fktable()
wp_t3 = wp[:, t3_index, :]
wd_t3 = wd[:, t3_index, :]

entry_p_rel = merged_df["entry_p"].to_numpy() - 1
entry_d_rel = merged_df["entry_d"].to_numpy() - 1
W = wp_t3[entry_p_rel] - wd_t3[entry_d_rel]  # shape (n_data, n_grid)

# Save xgrid for later normalization
xgrid = fk_p.xgrid.copy()  # shape (n_grid,)
print(np.min(xgrid), np.max(xgrid), len(xgrid))


idx_p_merge = merged_df["idx_p"].to_numpy()  # length = N (number of matched points)
idx_d_merge = merged_df["idx_d"].to_numpy()  # length = N (same N)

# cov_full is (Np + Nd) x (Np + Nd)
n_p = len(df_p)
c_yy = project_pd_cov_to_difference(
    cov_full=cov_full,
    idx_p_merge=idx_p_merge,
    idx_d_merge=idx_d_merge,
    n_p=n_p,
)


# Add jitter until positive-definite
jitter = 1e-6 * np.mean(np.diag(c_yy))
for _ in range(10):
    try:
        np.linalg.cholesky(c_yy)
        break
    except np.linalg.LinAlgError:
        c_yy += np.eye(c_yy.shape[0]) * jitter
        jitter *= 10
else:
    msg = "Covariance matrix not positive-definite"
    raise RuntimeError(msg)

# Compute reference for closure test, this should be the f0/y0
pdf0 = pdfset.mkPDF(0)
Q0 = fk_p.Q0
xt3_true = np.zeros_like(xgrid)

# T_3 = (u + ubar) - (d + dbar)
for i, x in enumerate(xgrid):
    u = pdf0.xfxQ(2, x, Q0)
    ub = pdf0.xfxQ(-2, x, Q0)
    d = pdf0.xfxQ(1, x, Q0)
    db = pdf0.xfxQ(-1, x, Q0)
    xt3_true[i] = (u + ub) - (d + db)

t3 = (
    xt3_true / xgrid
)  # Or here we can just input the true function directly for T3 or xT3

t3_ref_int = np.trapz(xt3_true / xgrid, xgrid)  # noqa: NPY201
# print("q0:", (q2_vals))
print("q2_vals shape:", (q2_vals.shape))
print("xgrid shape: ", xgrid.shape)
print("xt3_true shape; ", xt3_true.shape)
y_theory = W @ (xt3_true)  # shape (N,)
y_t3_theory = W @ (t3)  # shape (N,)

print("y theory shape: ", y_theory.shape)
# y_test, load different FK table to change basis to F3

# L1 data
rng = np.random.default_rng(seed=451)  # previously used seed
# rng = np.random.default_rng(seed=357) # best seed
noise = rng.multivariate_normal(mean=np.zeros(len(y_theory)), cov=c_yy)

y_pseudo = y_theory + noise

# generate C_T0 matrix
c_t0_yy, cov_t0_full, t0_predictions = build_c_t0_yy_from_validphys(
    lcd_p=lcd_p,
    lcd_d=lcd_d,
    idx_p_merge=idx_p_merge,
    idx_d_merge=idx_d_merge,
    n_p=n_p,
    y_vals=y_vals,
    y_theory=y_theory,
)

t0_pred_p, t0_pred_d = t0_predictions

print("c_t0_yy shape; ", c_t0_yy.shape)

# L2 data (MC replicas around L1 data)
n_l2_replicas = 100
rng_l2 = np.random.default_rng(seed=452)

# use same cov as for L1, experimental unc matrix in this case
noise_l2 = rng_l2.multivariate_normal(
    mean=np.zeros(len(y_pseudo)),
    cov=c_yy,
    size=n_l2_replicas,
    )

y_l2 = y_pseudo + noise_l2

print("y_l2 shape: ", y_l2.shape)


# Saving data
theoryid = str(theoryid)

np.savez(
    f"Dataset/data_{theoryid}.npz",
    q2_vals=q2_vals,
    y_vals=y_vals,
    y_pseudo=y_pseudo,
    W=W,
    xgrid=xgrid,
    y_theory=y_theory,
    xt3_true=xt3_true,
    c_yy=c_yy,
    c_t0_yy=c_t0_yy,
    cov_t0_full=cov_t0_full,
    t0_pred_p=t0_pred_p,
    t0_pred_d=t0_pred_d,
    kinematics=kinematics,
    y_l2=y_l2,
    x_data=x_data,
)
# print(f"Saved as Dataset/data_{theoryid}.npz")