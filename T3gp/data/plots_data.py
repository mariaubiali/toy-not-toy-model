"""T3 Data Comparison Script (+ Pseudo-data).

Compares author-provided prepared data,
our own real-data-processed version,
AND a closure-test "pseudo-data" version where
we generate y_pseudo = W_our @ T3_ref_our + Gaussian noise.

Usage:
    python compare_t3_data.py
"""

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# ------------- #
# 1. File Paths #
# ------------- #
auth_data_path = "../../T3nn/data/prepared_data/"
self_data_path = "Dataset/"

theoryid = 208
theoryid_str = str(theoryid)

# -------------- #
# 2. Load Both   #
# -------------- ##
# -- Author's processed data --
y_auth = np.load(auth_data_path + "data.npy")
Cy_auth = np.load(auth_data_path + "Cy.npy")
kin_auth = np.load(auth_data_path + "kin.npy")
FK_auth = np.load(auth_data_path + "FK.npy")
xgrid_auth = np.load(auth_data_path + "fk_grid.npy")
NNPDF40_auth = np.load(auth_data_path + "NNPDF40.npy")

T3_ref_auth = NNPDF40_auth[6 * 50 : 7 * 50]

print(y_auth.shape)

# --- Load generated data ---

data = np.load(f"Dataset/data_{theoryid}.npz")

q2_vals = data["q2_vals"]
y_vals = data["y_vals"]
y_pseudo = data["y_pseudo"]
W = data["W"]
xgrid = data["xgrid"]
y_theory = data["y_theory"]
xt3_true = data["xt3_true"]
c_yy = data["c_yy"]
kinematics = data["kinematics"]

# ---- Plotting -----
# ------------- #
# 4. Comparison #
# ------------- #

# == (1) Kinematic coverage ==
fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
axs[0].scatter(kin_auth[:, 0], kin_auth[:, 1], c=y_auth, cmap="coolwarm", s=15)
axs[0].set(
    xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Author: kinematics"
)
axs[1].scatter(kinematics[:, 0], kinematics[:, 1], c=y_vals, cmap="coolwarm", s=15)
axs[1].set(
    xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Ours: kinematics"
)
sc = axs[2].scatter(
    kinematics[:, 0], kinematics[:, 1], c=y_pseudo, cmap="coolwarm", s=15
)
axs[2].set(
    xscale="log", yscale="log", xlabel="x", ylabel="Q²", title="Pseudo: kinematics"
)
plt.colorbar(sc, ax=axs[2], label="F2p-F2d or pseudo-y")
plt.tight_layout()
plt.show()

# == (2) y vector ==
plt.figure(figsize=(8, 4))
plt.plot(y_auth, ".", label="Author", alpha=0.8)
plt.plot(y_vals, ".", label="Ours (real data)", alpha=0.8)
plt.plot(y_pseudo, ".", label="Pseudo-data", alpha=0.7)
plt.xlabel("Matched Data Index")
plt.ylabel("F2p-F2d or pseudo-y")
plt.title("Fp-Fd: Author vs Ours vs Pseudo")
plt.legend()
plt.show()

# == (3) Covariance: diagonal and full matrix ==
plt.figure(figsize=(8, 4))
plt.plot(np.diag(Cy_auth), label="Author diag(C)")
plt.plot(np.diag(c_yy), label="Ours diag(C)")
plt.xlabel("Index")
plt.ylabel("Variance")
plt.title("Covariance Diagonal: Author vs Ours")
plt.legend()
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.imshow(Cy_auth, aspect="auto", origin="lower")
plt.title("Author: Covariance")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(c_yy, aspect="auto", origin="lower")
plt.title("Ours: Covariance")
plt.colorbar()
plt.suptitle("Covariance Matrices (full)")
plt.tight_layout()
plt.show()

# == (4) FK Table comparison ==
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(FK_auth, aspect="auto", origin="lower")
plt.title("Author FK (248 x 50)")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(W, aspect="auto", origin="lower")
plt.title("Ours W = FKT3p - FKT3d (N x 50)")
plt.colorbar()
plt.suptitle("FK Table Comparison")
plt.tight_layout()
plt.show()

# == Per-row correlation ==
min_rows = min(FK_auth.shape[0], W.shape[0])
corrs = [np.corrcoef(FK_auth[i], W[i])[0, 1] for i in range(min_rows)]
plt.figure()
plt.plot(corrs)
plt.xlabel("Matched Data Index")
plt.ylabel("Correlation (Author FK vs Our W)")
plt.title("Per-row correlation (should be ~1)")
plt.show()

# == (5) x-grid ==
plt.figure()
plt.plot(xgrid_auth, label="Author xgrid")
plt.plot(xgrid, "--", label="Our xgrid")
plt.xlabel("x-grid index")
plt.ylabel("x")
plt.title("x-grid: Author vs Ours")
plt.legend()
plt.show()

# == (6) NNPDF T3 reference ==
plt.figure()
plt.plot(xgrid_auth, T3_ref_auth, label="Author T3_ref (NNPDF4.0)")
plt.plot(xgrid, xt3_true, "--", label="Ours T3_ref (LHAPDF)")
plt.xlabel("x")
plt.ylabel("T₃(x)")
plt.title("NNPDF T₃: Author vs Ours")
plt.legend()
plt.show()

plt.figure()
plt.plot(xgrid_auth, T3_ref_auth, label="Author T3_ref (NNPDF4.0)")
plt.plot(xgrid, xt3_true, "--", label="Ours T3_ref (LHAPDF)")
plt.xlabel("x")
plt.xscale("log")
plt.ylabel("T₃(x)")
plt.title("NNPDF T₃: Author vs Ours")
plt.legend()
plt.show()

# == (7) FK @ T3_ref convolution ==

# (a) Author: y_pred = FK @ T3_ref_auth
y_pred_auth = FK_auth @ T3_ref_auth
plt.figure()
plt.scatter(y_pred_auth, y_auth, s=18, alpha=0.7, label="Author: y_pred vs y")
plt.plot([y_auth.min(), y_auth.max()], [y_auth.min(), y_auth.max()], "k--", alpha=0.5)
plt.xlabel("y_pred (FK·T3_ref)")
plt.ylabel("y (data)")
plt.title("Author: FK convolution")
plt.legend()
plt.show()
# (b) Ours: y_pred = W_our @ T3_ref_our
y_pred_our = W @ xt3_true
plt.figure()
plt.scatter(y_pred_our, y_vals, s=18, alpha=0.7, label="Ours: y_pred vs y")
plt.plot([y_vals.min(), y_vals.max()], [y_vals.min(), y_vals.max()], "k--", alpha=0.5)
plt.xlabel("y_pred (W·T3_ref)")
plt.ylabel("y (data)")
plt.title("Ours: FK convolution")
plt.legend()
plt.show()
# (c) Pseudo: y_pred = W_our @ T3_ref_our vs y_pseudo
plt.figure()
plt.scatter(y_pred_our, y_pseudo, s=18, alpha=0.7, label="Pseudo: y_pred vs y_pseudo")
plt.plot(
    [y_pseudo.min(), y_pseudo.max()], [y_pseudo.min(), y_pseudo.max()], "k--", alpha=0.5
)
plt.xlabel("y_pred (W·T3_ref)")
plt.ylabel("y_pseudo (pseudo-data)")
plt.title("Pseudo: FK convolution")
plt.legend()
plt.show()
