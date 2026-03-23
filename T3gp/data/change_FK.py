#!/usr/bin/env python3

import numpy as np

INPUT_NPZ = "Dataset/data_208.npz"

OUTPUT_20 = "Dataset/data_208_proj5.npz"
OUTPUT_50 = "Dataset/data_208_proj50.npz"
OUTPUT_200 = "Dataset/data_208_proj200.npz"


def build_interp_matrix_logx(x_target, x_source):
    """
    Build interpolation matrix P such that

        f(x_target) ~= P @ f(x_source)

    using piecewise linear interpolation in log(x).

    x_target : grid where function is to be reconstructed
    x_source : grid where function coefficients live
    """
    x_target = np.asarray(x_target, dtype=float)
    x_source = np.asarray(x_source, dtype=float)

    lx_target = np.log(x_target)
    lx_source = np.log(x_source)

    P = np.zeros((len(x_target), len(x_source)))

    for i, xt in enumerate(lx_target):
        if xt <= lx_source[0]:
            P[i, 0] = 1.0
        elif xt >= lx_source[-1]:
            P[i, -1] = 1.0
        else:
            j = np.searchsorted(lx_source, xt) - 1
            x0, x1 = lx_source[j], lx_source[j + 1]
            w1 = (xt - x0) / (x1 - x0)
            w0 = 1.0 - w1
            P[i, j] = w0
            P[i, j + 1] = w1

    return P


def project_xt3_to_new_grid(xgrid_old, xt3_old, xgrid_new):
    """
    Given xt3 on the old grid, find xt3 on the new grid such that

        xt3_old ~= P_old_from_new @ xt3_new

    in least-squares sense.
    """
    if len(xgrid_new) <= len(xgrid_old):
        P_old_from_new = build_interp_matrix_logx(xgrid_old, xgrid_new)
        xt3_new, *_ = np.linalg.lstsq(P_old_from_new, xt3_old, rcond=None)
    else:
        P_old_from_new = build_interp_matrix_logx(xgrid_old, xgrid_new)
        xt3_new = np.interp(np.log(xgrid_new), np.log(xgrid_old), xt3_old)
    return xt3_new, P_old_from_new


def print_diffs(label, ref, test):
    absdiff = np.abs(test - ref)
    reldiff = absdiff / np.maximum(np.abs(ref), 1e-14)
    print(f"\n{label}")
    print("max abs diff:", absdiff.max())
    print("mean abs diff:", absdiff.mean())
    print("max rel diff:", reldiff.max())
    print("mean rel diff:", reldiff.mean())


def save_projected_dataset(
    output_npz,
    xgrid_target,
    xgrid_bench,
    W_bench,
    xt3_bench,
    y_theory_bench,
    y_pseudo_bench,
    c_yy,
    q2_vals,
    y_vals,
    kinematics,
):
    # Reconstruct old-grid function from target-grid coefficients:
    # xt3_bench ~= P_bench_from_target @ xt3_target
    xt3_target, P_bench_from_target = project_xt3_to_new_grid(
        xgrid_bench, xt3_bench, xgrid_target
    )

    # Effective FK table in the target basis
    W_target = W_bench @ P_bench_from_target

    # Propagated theory prediction
    y_theory_target = W_target @ xt3_target

    # Reuse exact same pseudo-noise realization from benchmark
    noise = y_pseudo_bench - y_theory_bench
    y_pseudo_target = y_theory_target + noise

    np.savez(
        output_npz,
        q2_vals=q2_vals,
        y_vals=y_vals,
        y_pseudo=y_pseudo_target,
        W=W_target,
        xgrid=xgrid_target,
        y_theory=y_theory_target,
        xt3_true=xt3_target,
        c_yy=c_yy,
        kinematics=kinematics,
    )

    return {
        "xgrid": xgrid_target,
        "W": W_target,
        "xt3_true": xt3_target,
        "y_theory": y_theory_target,
        "y_pseudo": y_pseudo_target,
    }


# -----------------------------
# 1. Load benchmark dataset
# -----------------------------
data = np.load(INPUT_NPZ)

xgrid_bench = data["xgrid"]
W_bench = data["W"]
xt3_bench = data["xt3_true"]
y_theory_bench = data["y_theory"]
y_pseudo_bench = data["y_pseudo"]
c_yy = data["c_yy"]
kinematics = data["kinematics"]
q2_vals = data["q2_vals"]
y_vals = data["y_vals"]

print("Loaded benchmark:")
print("xgrid length:", len(xgrid_bench))
print("W shape:", W_bench.shape)
print("xt3_true shape:", xt3_bench.shape)
print("y_theory shape:", y_theory_bench.shape)
print("c_yy shape:", c_yy.shape)

# -----------------------------
# 2. Define target grids
# -----------------------------
xgrid_20 = np.geomspace(xgrid_bench.min(), xgrid_bench.max(), 5)

# For the 50-point check, use the SAME benchmark grid exactly
xgrid_50 = xgrid_bench.copy()

xgrid_200 = np.geomspace(xgrid_bench.min(), xgrid_bench.max(), 200)

print("\nTarget grids:")
print("20-point grid length:", len(xgrid_20))
print("50-point grid length:", len(xgrid_50))
print("200-point grid length:", len(xgrid_200))

# -----------------------------
# 3. Build and save projected datasets
# -----------------------------
proj5 = save_projected_dataset(
    OUTPUT_20,
    xgrid_20,
    xgrid_bench,
    W_bench,
    xt3_bench,
    y_theory_bench,
    y_pseudo_bench,
    c_yy,
    q2_vals,
    y_vals,
    kinematics,
)

# proj50 = save_projected_dataset(
#     OUTPUT_50,
#     xgrid_50,
#     xgrid_bench,
#     W_bench,
#     xt3_bench,
#     y_theory_bench,
#     y_pseudo_bench,
#     c_yy,
#     q2_vals,
#     y_vals,
#     kinematics,
# )

# proj200 = save_projected_dataset(
#     OUTPUT_200,
#     xgrid_200,
#     xgrid_bench,
#     W_bench,
#     xt3_bench,
#     y_theory_bench,
#     y_pseudo_bench,
#     c_yy,
#     q2_vals,
#     y_vals,
#     kinematics,
# )

# -----------------------------
# 4. Diagnostics
# -----------------------------
# print_diffs("20 vs benchmark theory", y_theory_bench, proj20["y_theory"])
# print_diffs("50 vs benchmark theory", y_theory_bench, proj50["y_theory"])
# print_diffs("200 vs benchmark theory", y_theory_bench, proj200["y_theory"])

# noise_bench = y_pseudo_bench - y_theory_bench
# noise_20 = proj20["y_pseudo"] - proj20["y_theory"]
# noise_50 = proj50["y_pseudo"] - proj50["y_theory"]
# noise_200 = proj200["y_pseudo"] - proj200["y_theory"]

# print("\nNoise reuse checks:")
# print("max |noise_20 - noise_bench| :", np.max(np.abs(noise_20 - noise_bench)))
# print("max |noise_50 - noise_bench| :", np.max(np.abs(noise_50 - noise_bench)))
# print("max |noise_200 - noise_bench|:", np.max(np.abs(noise_200 - noise_bench)))

# print("\nSaved:")
# print(" ", OUTPUT_20)
# print(" ", OUTPUT_50)
# print(" ", OUTPUT_200)