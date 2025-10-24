# %% [markdown]
# # t3_BSM_Comparison
#
# This notebook-style script performs a self-contained closure test and BSM Wilson-coefficient
# reconstruction in the non-singlet PDF channel $t_3(x)$.

# %%
# %%
"""t3_BSM_Comparison."""

# %%
# --- Imports & Setup ---
from __future__ import annotations

from pathlib import Path

import lhapdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from validphys.api import API
from validphys.fkparser import load_fktable
from validphys.loader import Loader

# Device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories for outputs
model_state_dir = Path("model_states")
model_state_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

image_dir = Path("images")
# %% [markdown]
# ## Data Loading & Preprocessing — Part 1
#
# Fetch raw BCDMS $F_2^p$ and $F_2^d$ tables, rename columns for clarity, and compute
# the difference $y = F_2^p - F_2^d$.


# %%
# DATA LOADING & PREPROCESSING—PART 1: FETCH RAW TABLES
# ------------------------------------------------------------------------------
logger.info("Loading BCDMS F2 data from validphys API...")

inp_p = {
    "dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_P_EM-F2", "variant": "legacy"},
    "use_cuts": "internal",
    "theoryid": 208,
}
inp_d = {
    "dataset_input": {"dataset": "BCDMS_NC_NOTFIXED_D_EM-F2", "variant": "legacy"},
    "use_cuts": "internal",
    "theoryid": 208,
}

lcd_p = API.loaded_commondata_with_cuts(**inp_p)
lcd_d = API.loaded_commondata_with_cuts(**inp_d)

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
    .assign(idx_p=lambda df: df.index)
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
    .assign(idx_d=lambda df: df.index)
)


# Merge on (x, q2) to form F2_p - F2_d
mp = 0.938
mp2 = mp**2
merged_df = df_p.merge(df_d, on=["x", "q2"], suffixes=("_p", "_d")).assign(
    y_val=lambda df: (df["F2_p"] - df["F2_d"]),
    w2=lambda df: df["q2"] * (1 - df["x"]) / df["x"] + mp2,
)

# Extract q2_vals and y_real for later use
q2_vals = merged_df["q2"].to_numpy()
y_real = merged_df["y_val"].to_numpy()
# %% [markdown]
# ## Data Loading & Preprocessing — Part 2
#
# Build FK tables for proton and deuteron, extract the t3 channel (flavor index 2), and
# form the convolution matrix $W$.


# %%
# DATA LOADING & PREPROCESSING—PART 2: BUILD FK TABLES & W
# ------------------------------------------------------------------------------
logger.info("Building FK tables and computing convolution matrix W for t3 channel...")

t3_index = 2  # flavor index in FK table
loader = Loader()
fk_p = load_fktable(loader.check_fktable(setname="BCDMSP", theoryID=208, cfac=()))
fk_d = load_fktable(loader.check_fktable(setname="BCDMSD", theoryID=208, cfac=()))

wp = fk_p.get_np_fktable()  # shape (n_data_fk, n_flav, n_grid)
wd = fk_d.get_np_fktable()
wp_t3 = wp[:, t3_index, :]
wd_t3 = wd[:, t3_index, :]

entry_p_rel = merged_df["entry_p"].to_numpy() - 1
entry_d_rel = merged_df["entry_d"].to_numpy() - 1
W = wp_t3[entry_p_rel] - wd_t3[entry_d_rel]  # shape (n_data, n_grid)

# Save xgrid for later normalization
xgrid = fk_p.xgrid.copy()  # shape (n_grid,)
# %% [markdown]
# ## Data Loading & Preprocessing — Part 3
#
# Construct the covariance matrix $C_{yy}$ for $y$, extract sub-blocks for proton and
# deuteron, symmetrize, and add jitter until positive-definite.

# %%
# DATA LOADING & PREPROCESSING—PART 3: COMPUTE C_YY & ITS INVERSE
# ------------------------------------------------------------------------------
logger.info("Building covariance matrix c_yy for y = F2_p - F2_d...")

params_cov = {
    "dataset_inputs": [inp_p["dataset_input"], inp_d["dataset_input"]],
    "use_cuts": "internal",
    "theoryid": 208,
}
cov_full = API.dataset_inputs_covmat_from_systematics(**params_cov)

# Suppose merged_df has columns idx_p and idx_d (these were created earlier in your preprocessing)
idx_p_merge = merged_df["idx_p"].to_numpy()  # length = N (number of matched points)
idx_d_merge = merged_df["idx_d"].to_numpy()  # length = N (same N)

# cov_full is (Np + Nd) x (Np + Nd), so:
n_p = len(df_p)
# Extract the proton-proton, deuteron-deuteron, and proton-deuteron sub-blocks:
c_pp = cov_full[:n_p, :n_p]  # shape = (Np, Np)
c_dd = cov_full[n_p:, n_p:]  # shape = (Nd, Nd)
c_pd = cov_full[:n_p, n_p:]  # shape = (Np, Nd)

# Now restrict each block to only those rows/cols that appear in merged_df:
c_pp_sub = c_pp[np.ix_(idx_p_merge, idx_p_merge)]  # (N, N)
c_dd_sub = c_dd[np.ix_(idx_d_merge, idx_d_merge)]  # (N, N)
c_pd_sub = c_pd[np.ix_(idx_p_merge, idx_d_merge)]  # (N, N)


c_yy = c_pp_sub + c_dd_sub - 2 * c_pd_sub

# Make sure it's exactly symmetric:
c_yy = 0.5 * (c_yy + c_yy.T)


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

# %% [markdown]
# ## Compute Reference t3 for Closure
#
# Load the central PDF, compute the true non-singlet $x t_3(x)$, and generate pseudo-data
# by adding correlated noise.

# %%
# DATA LOADING & PREPROCESSING—PART 4: COMPUTE t3_REF_NORM FOR CLOSURE
# ------------------------------------------------------------------------------
logger.info("Computing reference t3 (t3_ref_norm) for closure test...")

pdfset = lhapdf.getPDFSet("NNPDF40_nnlo_as_01180")
pdf0 = pdfset.mkPDF(0)
Q0 = fk_p.Q0
xt3_true = np.zeros_like(xgrid)

for i, x in enumerate(xgrid):
    u = pdf0.xfxQ(2, x, Q0)
    ub = pdf0.xfxQ(-2, x, Q0)
    d = pdf0.xfxQ(1, x, Q0)
    db = pdf0.xfxQ(-1, x, Q0)
    xt3_true[i] = (u - ub) - (d - db)


t3 = xt3_true / xgrid

t3_ref_int = np.trapz(xt3_true / xgrid, xgrid)  # noqa: NPY201


y_theory = W @ (xt3_true)  # shape (N,)

rng = np.random.default_rng(seed=451)  # you can set seed if you want reproducible “data”
noise = rng.multivariate_normal(mean=np.zeros(len(y_theory)), cov=c_yy)

y_pseudo = y_theory + noise
# %% [markdown]
# ## Preliminary Data Plots
#
# 1. Scatter real vs theory and pseudo vs theory
# 2. Heatmap of mean difference
# 3. Theory/data ratio with error bars
# 4. Kinematic coverage subplots


# %%
# PRELIM DATA PLOTS
plt.figure()

# 1) Real data vs. Theory (open blue circles)
plt.scatter(
    y_theory,
    y_real,
    s=24,
    alpha=0.7,
    facecolors="none",
    edgecolors="C0",
    label=r"Real Data: $y_{data} = F_{2}^{p} - F_{2}^{d}$",
)

# 2) Pseudo-data vs. Theory (filled orange dots)
plt.scatter(
    y_theory,
    y_pseudo,
    s=18,
    alpha=0.6,
    color="C1",
    label=r"Pseudo-Data: $y_{pseudo} = W\,xt_{3}^{NNPDF} + \eta$",
)

# 3) Diagonal y = x (gray dashed line)
mn = min(y_theory.min(), y_real.min(), y_pseudo.min())
mx = max(y_theory.max(), y_real.max(), y_pseudo.max())
plt.plot(
    [mn, mx],
    [mn, mx],
    linestyle="--",
    color="gray",
    alpha=0.5,
    label=r"$y_{theory} = y_{observed}$",
)

# 4) Labels and Title (all math-text in "$...$")
plt.xlabel(
    r"$y_{theory} = [\,W \cdot x\,t_{3}(x)\,]_{NNPDF40}$",
    fontsize=14,
)
plt.ylabel(r"$y_{observed}$", fontsize=14)

plt.title(
    r"Comparison of Real vs. Pseudo-Data for $F_{2}^{p} - F_{2}^{d}$",
)

plt.legend(loc="upper right", frameon=True, edgecolor="k")
plt.grid(alpha=0.2)
plt.savefig(image_dir / "real_vs_theory.png", bbox_inches="tight")
plt.show()


# %%
# Heatmap Plot
merged_df["y_theory"] = y_theory

pivot_real = (
    merged_df.pivot_table(index="q2", columns="x", values="y_val", aggfunc="mean")
    .sort_index(axis=0)
    .sort_index(axis=1)
)
pivot_theory = (
    merged_df.pivot_table(index="q2", columns="x", values="y_theory", aggfunc="mean")
    .sort_index(axis=0)
    .sort_index(axis=1)
)
pivot_diff = pivot_real - pivot_theory


x_vals = pivot_real.columns.to_numpy()  # (N_x,)
q2_vals = pivot_real.index.to_numpy()  # (N_q2,)
X_grid, Y_grid = np.meshgrid(x_vals, q2_vals)


fig, ax = plt.subplots(figsize=(7, 6))

pcm = ax.pcolormesh(
    X_grid,
    Y_grid,
    pivot_diff.values,
    shading="auto",
    cmap="RdBu_r",  # diverging colormap is often useful for “difference”
    vmin=-np.max(np.abs(pivot_diff.values)),  # center zero at white
    vmax=np.max(np.abs(pivot_diff.values)),
)

cbar = fig.colorbar(pcm, ax=ax, label=r"$\langle\,y_{\rm data} - y_{\rm theory}\rangle$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(r"Mean Difference: $\langle\,y_{\rm data} - y_{\rm theory}\rangle$")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$Q^2\,[\mathrm{GeV}^2]$")

plt.savefig(image_dir / "mean_difference_theory.png", bbox_inches="tight")
plt.show()

# %%
# ? Theory Comparison

# Compute sigma_i = sqrt(diagonal(C_yy))_i  divided by y_real_i
sigma = np.abs(np.sqrt(np.diag(c_yy)) / y_real)

# Make an index array to place points on the x-axis
x_idx = np.arange(len(y_theory))  # 0, 1, 2, … N-1
ref = np.ones_like(y_theory)  # reference = 1 for “data/theory = 1”


plt.figure(figsize=(20, 5))
plt.errorbar(
    x_idx,
    ref,
    sigma,
    fmt="none",
    ecolor="gray",
    alpha=0.5,
    label=r"Data uncertainty $( \frac{\sigma_i}{y_i})$",
)
plt.scatter(
    x_idx,
    y_theory / y_real,
    marker="*",
    c="red",
    label="Theory / Data",
)

plt.ylim([0.1, 2.5])
plt.xlabel("Data point index (i)")
plt.ylabel(r"$\frac{y_{theory}}{y_{data}}$")
plt.title(r"Comparison of $y_{theory}$ vs.\ $y_{data}$ (with relative errors)")
plt.legend(loc="upper right")
plt.grid(alpha=0.3)
plt.savefig(image_dir / "data_theory_error_comp.png", bbox_inches="tight")
plt.show()

# %%
# Kinematic Plot
fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

# Proton subplot
ax_p.scatter(
    df_p["x"],
    df_p["q2"],
    marker="o",
    c="C0",
    label=r"$F_2^p$",
    alpha=0.7,
)
ax_p.set_xscale("log")
ax_p.set_yscale("log")

ax_p.set_xlabel(r"$x$")
ax_p.set_ylabel(r"$Q^2\ \mathrm{[GeV^2]}$")
ax_p.set_title("BCDMS $F_2^p$")
ax_p.grid(which="both", alpha=0.3)

# Deuteron subplot
ax_d.scatter(
    df_d["x"],
    df_d["q2"],
    marker="s",
    c="C1",
    label=r"$F_2^d$",
    alpha=0.7,
)
ax_d.set_xscale("log")
ax_d.set_yscale("log")

ax_d.set_xlabel(r"$x$")
# Only include ylabel on the left subplot to avoid redundancy
ax_d.set_title("BCDMS $F_2^d$")
ax_d.grid(which="both", alpha=0.3)

plt.suptitle("Kinematic Coverage of BCDMS $F_2^p$ and $F_2^d$", y=1.02)
plt.savefig(image_dir / "kineamtic_coverage.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Neural Network Definitions
#
# Define `T3Net` (no BSM) and `T3NetWithC` (with single parameter C) with preprocessing layers
# for $x^\alpha (1-x)^\beta$

# %%
# NEURAL NETWORK MODEL DEFINITION
# ------------------------------------------------------------------------------


class T3Net(nn.Module):
    """Neural network for non-singlet PDF t₃(x) with preprocessing x^alpha (1-x)^beta."""

    def __init__(
        self,
        n_hidden: int,
        n_layers: int = 3,
        init_alpha: float = 1.0,
        init_beta: float = 3.0,
        dropout: float = 0.2,
    ) -> None:
        """Create T3 Net."""
        super().__init__()
        # Log-parametrization for alpha, beta
        self.logalpha = nn.Parameter(torch.log(torch.tensor(init_alpha)))
        self.logbeta = nn.Parameter(torch.log(torch.tensor(init_beta)))

        # Build MLP: [Linear → Tanh → BatchNorm] x (n_layers), ending in Linear
        layers: list[nn.Module] = [nn.Linear(1, n_hidden), nn.Tanh(), nn.BatchNorm1d(n_hidden)]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.Tanh(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(n_hidden, 1))  # final raw output
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → returns x · t₃_unc(x) ≥ 0.

        raw = self.net(x) is unconstrained; apply SoftPlus to ensure nonnegativity.
        Multiply by x^alpha (1-x)^beta to impose endpoints behavior.
        """
        raw = self.net(x).squeeze()  # shape (N_grid,)
        pos = F.softplus(raw)  # shape (N_grid,), enforces ≥ 0

        alpha = torch.exp(self.logalpha).clamp(min=1e-3)
        beta = torch.exp(self.logbeta).clamp(min=1e-3)
        x_ = x.squeeze().clamp(min=1e-6, max=1 - 1e-6)

        pre = x_.pow(alpha) * (1.0 - x_).pow(beta)  # shape (N_grid,)
        return pre * pos  # returns x · t₃_unc(x)


class T3NetWithC(nn.Module):
    """Neural network for x·t₃(x) plus a single BSM parameter C."""

    def __init__(
        self,
        n_hidden: int,
        n_layers: int = 3,
        init_alpha: float = 1.0,
        init_beta: float = 3.0,
        dropout: float = 0.2,
    ) -> None:
        """Init our T3Net with additional BSM C Param."""
        super().__init__()
        # 1) instantiate the original T3Net
        self.base = T3Net(
            n_hidden=n_hidden,
            n_layers=n_layers,
            init_alpha=init_alpha,
            init_beta=init_beta,
            dropout=dropout,
        )
        # 2) expose logalpha/logbeta so that `model.logalpha` still works
        self.logalpha = self.base.logalpha
        self.logbeta = self.base.logbeta

        # 3) add a single learnable scalar C (initialized at 0)
        self.C = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns f_raw = x · t₃_unc(x) (shape = n_grid,).

        The BSM correction factor (1 + C·K) is applied downstream in the training loop.
        """
        # simply delegate to the base network
        return self.base(x).squeeze()  # shape = (n_grid,)


# %% [markdown]
# ## Ansatz Functions & Configuration
#
# Build two ansatz shapes (K1, K2), normalize them, and prepare a `config` dict for standard
# fits and sensitivity scans.


# %%
# DEFINE BASE ANSATZ FUNCTIONS (NORMALIZED TO UNIT AMPLITUDE) AND BUILD CONFIG DICTIONARY
# INCLUDING ORIGINAL FITS + SENSITIVITY SCANS
q2_vals = merged_df["q2"].to_numpy()  # (n_data,)
x_vals_data = merged_df["x"].to_numpy()  # (n_data,)
Q2_min = q2_vals.min()

# Raw (unnormalized) shapes
K1_raw = (q2_vals - Q2_min) ** 2
K2_raw = x_vals_data * (1.0 - x_vals_data) * (q2_vals - Q2_min)

# Normalize so max(|K_raw|)=1
K1_unit = K1_raw / np.max(np.abs(K1_raw))
K2_unit = K2_raw / np.max(np.abs(K2_raw))

# Convert to torch tensors
K_dict = {
    "ansatz1": torch.tensor(K1_unit, dtype=torch.float32, device=device),
    "ansatz2": torch.tensor(K2_unit, dtype=torch.float32, device=device),
}

C_trues = [0.0, 0.1, 1]


config = {
    # Original fits (no BSM)
    "fit_real_real": {
        "name": "Real-Data Fit",
        "input_key": "real_real",
        "n_hidden": 30,
        "n_layers": 3,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 500,
        "num_epochs": 5000,
        "n_replicas": 100,
        "lambda_sr": 0.0,
        "bsm": False,
        "ansatz": None,
        "C_true": 0.0,
    },
    "fit_pseudo_replica": {
        "name": "Pseudo-Replica Fit",
        "input_key": "pseudo_replica",
        "n_hidden": 30,
        "n_layers": 3,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 500,
        "num_epochs": 5000,
        "n_replicas": 100,
        "lambda_sr": 10000.0,
        "bsm": False,
        "ansatz": None,
        "C_true": 0.0,
    },
}

for ansatz_name in ["ansatz1", "ansatz2"]:
    for C_true in C_trues:
        cfg_key = f"sens_{ansatz_name}_C{C_true:.0e}"
        display = {
            "ansatz1": "Sensitivity Scan 1 (Q²² shape)",
            "ansatz2": "Sensitivity Scan 2 (x(1-x)Q² shape)",
        }[ansatz_name]
        cfg_name = f"{display}, $C_{{true}}$={C_true:.0e}"
        config[cfg_key] = {
            "name": cfg_name,
            "input_key": "pseudo_replica",  # always pseudo-data for sensitivity scans
            "n_hidden": 30,
            "n_layers": 3,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "patience": 500,
            "num_epochs": 5000,
            "n_replicas": 100,
            "lambda_sr": 10000.0,
            "bsm": True,
            "ansatz": ansatz_name,
            "C_true": C_true,
        }
# %% [markdown]
# ## Training Loop
#
# For each config and replica:
# 1. Split train/validation
# 2. Prepare y-data (with or without BSM injection)
# 3. Build covariance inverses
# 4. Initialize model & optimizer
# 5. Train with early stopping & logging

# %%
# TRAINING
n_data = W.shape[0]
n_grid = xgrid.shape[0]
W_torch = torch.tensor(W, dtype=torch.float32, device=device)
x_torch = torch.tensor(xgrid, dtype=torch.float32).unsqueeze(1).to(device)


all_results = []


for cfg_key, cfg in config.items():
    input_key = cfg["input_key"]
    n_hidden = cfg["n_hidden"]
    n_layers = cfg["n_layers"]
    dropout = cfg["dropout"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    patience = cfg["patience"]
    num_epochs = cfg["num_epochs"]
    n_replicas = cfg["n_replicas"]
    lambda_sr = cfg["lambda_sr"]
    is_bsm = cfg["bsm"]
    ansatz_name = cfg["ansatz"]
    C_true = cfg["C_true"]
    display_name = cfg["name"]

    for replica in range(n_replicas):
        # ─── 5.a) Split train/validation indices ───
        torch.manual_seed(replica * 1234)
        idx_all = np.arange(n_data)
        train_idx, val_idx = train_test_split(idx_all, test_size=0.2, random_state=replica * 1000)

        # ─── 5.b) Prepare y-input (with or without BSM injection) ───
        rng = np.random.default_rng(seed=replica * 451)
        y_real_replica = rng.multivariate_normal(y_real, c_yy)
        y_pseudo_replica = rng.multivariate_normal(y_theory, c_yy)

        if is_bsm:
            K_torch = K_dict[ansatz_name]
            y_theory_bsm = (W @ t3) * (1.0 + C_true * K_torch.cpu().numpy())
            y_select = rng.multivariate_normal(y_theory_bsm, c_yy)
        else:
            y_select = {"real_real": y_real.copy(), "pseudo_replica": y_pseudo_replica}[input_key]

        y_torch = torch.tensor(y_select, dtype=torch.float32, device=device)

        # ─── 5.c) Build covariance inverses for train & val ───
        c_tr = c_yy[np.ix_(train_idx, train_idx)]
        c_val = c_yy[np.ix_(val_idx, val_idx)]
        Cinv_tr = torch.tensor(np.linalg.inv(c_tr), dtype=torch.float32, device=device)
        Cinv_val = torch.tensor(np.linalg.inv(c_val), dtype=torch.float32, device=device)

        # ─── 5.d) Initialize model & optimizer ───
        if is_bsm:
            model = T3NetWithC(
                n_hidden=n_hidden,
                n_layers=n_layers,
                init_alpha=1.0,
                init_beta=3.0,
                dropout=dropout,
            ).to(device)
        else:
            model = T3Net(
                n_hidden=n_hidden,
                n_layers=n_layers,
                init_alpha=1.0,
                init_beta=3.0,
                dropout=dropout,
            ).to(device)

        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        best_val_loss = float("inf")
        wait_counter = 0
        best_state = {}

        # ─── 5.e) TRAINING LOOP ───
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()

            f_raw = model(x_torch).squeeze()  # (n_grid,)
            y_pred_sm = W_torch.matmul(f_raw)  # (n_data,)

            if is_bsm:
                K_t = K_dict[ansatz_name]
                y_pred = y_pred_sm * (1.0 + model.C * K_t)
            else:
                y_pred = y_pred_sm

            resid_tr = y_pred[train_idx] - y_torch[train_idx]
            loss_chi2 = resid_tr @ (Cinv_tr.matmul(resid_tr))

            # Sum-rule penalty
            loss_sumrule = torch.tensor(0.0, device=device)
            if lambda_sr > 0.0:
                t3_unc = f_raw / x_torch.squeeze()
                I_mid = torch.trapz(t3_unc, x_torch.squeeze())
                loss_sumrule = lambda_sr * (I_mid - float(t3_ref_int)) ** 2

            loss_total = loss_chi2 + loss_sumrule
            loss_total.backward()
            optimizer.step()

            # ─── Validation χ² (no sum-rule penalty) ───
            model.eval()
            with torch.no_grad():
                f_raw_val = model(x_torch).squeeze()
                y_val_sm = W_torch[val_idx].matmul(f_raw_val)

                y_val_pred = y_val_sm * (1.0 + model.C * K_t[val_idx]) if is_bsm else y_val_sm

                resid_val = y_val_pred - y_torch[val_idx]
                loss_val = resid_val @ (Cinv_val.matmul(resid_val))
                val_chi2_pt = (loss_val / float(len(val_idx))).item()

            # ─── Early stopping ───
            if loss_val.item() < best_val_loss:
                best_val_loss = loss_val.item()
                wait_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait_counter += 1
                if wait_counter >= patience:
                    break

            # ─── Logging every 200 epochs ───
            if epoch % 200 == 0:
                if is_bsm:
                    logger.info(
                        f"{cfg_key} | Replica {replica} | Epoch {epoch:4d} | "
                        f"val χ²/pt = {val_chi2_pt:.4f} | C = {model.C.item():.3e}",
                    )
                else:
                    logger.info(
                        f"{cfg_key} | Replica {replica} | Epoch {epoch:4d} | "
                        f"val χ²/pt = {val_chi2_pt:.4f}",
                    )

        # ─── 5.f) Reload best state, compute final metrics on validation ───
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            f_raw_best = model(x_torch).squeeze()
            y_val_sm_best = W_torch[val_idx].matmul(f_raw_best)

            if is_bsm:
                y_val_pred_best = y_val_sm_best * (1.0 + model.C * K_t[val_idx])
            else:
                y_val_pred_best = y_val_sm_best

            resid_v = y_val_pred_best - y_torch[val_idx]
            chi2_val_final = float(resid_v @ (Cinv_val.matmul(resid_v)))
            chi2_pt = chi2_val_final / float(len(val_idx))

        alpha_val = float(torch.exp(model.logalpha).item())
        beta_val = float(torch.exp(model.logbeta).item())
        C_fit = float(model.C.item()) if is_bsm else float("nan")

        all_results.append(
            {
                "config_key": cfg_key,
                "config_name": display_name,
                "replica": replica,
                "alpha": alpha_val,
                "beta": beta_val,
                "C_true": C_true,
                "C_fit": C_fit,
                "chi2_pt": chi2_pt,
                "f_raw_best": f_raw_best,
            },
        )


df_results = pd.DataFrame(all_results)
df_results.to_pickle("training_results.pkl")


# %% [markdown]
# ## Results & Plotting
#
# - Filter by reduced chi squared: $\chi^2/{\rm pt}$ between 0.9 and 1.1
# - Plot real vs pseudo fits, sensitivity scans, alpha-beta ellipses, and C_fit
#   distributions in separate cells.


# %%
# Load in for plotting

df_raw = pd.read_pickle("training_results.pkl").reset_index()

# Then reduced Chi squared filter
df_results = df_raw.loc[lambda df: (df["chi2_pt"] < 1.1) & (df["chi2_pt"] > 0.9)]

# %%
# Real Pseudo Comp Plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True, sharey=True)
comparison_keys = ["fit_real_real", "fit_pseudo_replica"]
comparison_map = dict(zip(comparison_keys, axes))


display_names = {
    "fit_real_real": "Standard PDF fit",
    "fit_pseudo_replica": "Closure test: Simultaneous fit",
}

for cfg_key, ax in comparison_map.items():
    display_name = display_names[cfg_key]
    subset = df_results[df_results["config_key"] == cfg_key]

    # Stack replicas
    all_f = np.vstack(subset["f_raw_best"].values)
    mean_f = np.mean(all_f, axis=0)
    std_f = np.std(all_f, axis=0)

    # Metrics
    avg_sigma = np.mean(std_f)
    chi_vals = subset["chi2_pt"].to_numpy()
    mean_chi = np.mean(chi_vals)
    pct_within = 100 * np.sum(np.abs(xt3_true - mean_f) <= std_f) / len(xt3_true)

    # Plot ±1sigma band
    ax.fill_between(
        xgrid,
        mean_f - std_f,
        mean_f + std_f,
        color="C0",
        alpha=0.3,
        label=rf"$\pm\sigma={avg_sigma:.3f}$",
    )
    # Plot fit mean
    ax.plot(xgrid, mean_f, color="C0", linewidth=2, label=rf"$\chi^2/{{\rm pt}}={mean_chi:.2f}$")
    # Plot truth
    ax.plot(
        xgrid,
        xt3_true,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label=rf"Within $1\sigma={pct_within:.1f}\%$",
    )

    ax.set_title(display_name, fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$x\,t_{3}(x)$", fontsize=12)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(image_dir / "realvspseudofit.png", bbox_inches="tight")
plt.show()
# %%
# Sensitivtiy Scan plots

# Define ansatzes and C_true values
ansatzes = ["ansatz1", "ansatz2"]
C_trues = [0.0, 0.1, 1.0]
color_map = {0.0: "C0", 0.1: "C1", 1.0: "C2"}
doctitles = {"ansatz1": "Ansatz 1", "ansatz2": "Ansatz 2"}

# Create 2 rows x 3 cols with minimal spacing
grid_kw = {"wspace": 0.02, "hspace": 0.02}
fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(18, 10),
    sharex=True,
    sharey=True,
    gridspec_kw=grid_kw,
)

for i, ansatz in enumerate(ansatzes):
    for j, C_true in enumerate(C_trues):
        ax = axes[i, j]
        cfg_key = f"sens_{ansatz}_C{C_true:.0e}"
        subset = df_results[df_results["config_key"] == cfg_key]
        if subset.empty:
            ax.axis("off")
            continue

        # Prepare data
        all_f = np.vstack(subset["f_raw_best"].values)
        mean_f = np.mean(all_f, axis=0)
        std_f = np.std(all_f, axis=0)

        # Metrics
        avg_sigma = np.mean(std_f)
        chi_vals = subset["chi2_pt"].to_numpy()
        mean_chi = np.mean(chi_vals)

        # Plot uncertainty band
        ax.fill_between(
            xgrid,
            mean_f - std_f,
            mean_f + std_f,
            color=color_map[C_true],
            alpha=0.3,
            label=rf"$\pm\sigma={avg_sigma:.3f}$",
        )
        # Plot fit mean
        ax.plot(
            xgrid,
            mean_f,
            color=color_map[C_true],
            linewidth=2,
            label=rf"$\chi^2/{{\rm pt}}={mean_chi:.2f}$",
        )
        # Plot truth with percentage
        ax.plot(
            xgrid,
            xt3_true,
            color="k",
            linestyle="--",
            linewidth=1.5,
            label="Baseline",
        )

        # Column titles = plain values
        if i == 0:
            ax.set_title(rf"$C_{{true}}={C_true:g}$", fontsize=14)
        # Row labels = ansatz
        if j == 0:
            ax.set_ylabel(doctitles[ansatz], fontsize=14)
        # X-axis label on bottom row
        if i == len(ansatzes) - 1:
            ax.set_xlabel(r"$x$", fontsize=12)

        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

# Overall title, moved up
global_title = "Closure test: Wilson-coefficient reconstruction"
plt.suptitle(global_title, y=0.93, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.89])
plt.savefig(image_dir / "sensitivity_scan.png", bbox_inches="tight")
plt.show()


# %%
# Fixed-PDF vs BSM joint fit

ansatzes = ["ansatz1", "ansatz2"]
doctitles = {"ansatz1": "Ansatz 1", "ansatz2": "Ansatz 2"}
colors = {"Fixed-PDF": "C3", "Simultaneous fit": "C4"}

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), sharex=True, sharey=True)

# Precompute Fixed-PDF (C=0) statistics
subset_nb = df_results[df_results["config_key"] == "fit_pseudo_replica"]
all_f_nb = np.vstack(subset_nb["f_raw_best"].values)
mean_nb = np.mean(all_f_nb, axis=0)
std_nb = np.std(all_f_nb, axis=0)
avg_sigma_nb = np.mean(std_nb)
pct_within_nb = 100 * np.sum(np.abs(xt3_true - mean_nb) <= std_nb) / len(xt3_true)

for ax, ansatz in zip(axes, ansatzes):
    # Compute BSM joint-fit statistics for C_true=0
    cfg_bsm = f"sens_{ansatz}_C0e+00"
    subset_bsm = df_results[df_results["config_key"] == cfg_bsm]
    all_f_bsm = np.vstack(subset_bsm["f_raw_best"].values)
    mean_bsm = np.mean(all_f_bsm, axis=0)
    std_bsm = np.std(all_f_bsm, axis=0)
    avg_sigma_bsm = np.mean(std_bsm)
    pct_within_bsm = 100 * np.sum(np.abs(xt3_true - mean_bsm) <= std_bsm) / len(xt3_true)

    # Plot Fixed-PDF ±1sigma band and mean
    ax.fill_between(
        xgrid,
        mean_nb - std_nb,
        mean_nb + std_nb,
        color=colors["Fixed-PDF"],
        alpha=0.3,
    )
    ax.plot(
        xgrid,
        mean_nb,
        color=colors["Fixed-PDF"],
        linewidth=2,
        label=rf"Fixed-PDF: $\pm\sigma={avg_sigma_nb:.4f}$, {pct_within_nb:.1f}",
    )

    # Plot BSM ±1sigma band and mean
    ax.fill_between(
        xgrid,
        mean_bsm - std_bsm,
        mean_bsm + std_bsm,
        color=colors["Simultaneous fit"],
        alpha=0.3,
    )
    ax.plot(
        xgrid,
        mean_bsm,
        color=colors["Simultaneous fit"],
        linewidth=2,
        label=rf"Simultaneous fit: $\pm\sigma={avg_sigma_bsm:.4f}$, {pct_within_bsm:.1f}",
    )

    # Plot true t3(x)
    ax.plot(
        xgrid,
        xt3_true,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label=r"True $t_3(x)$",
    )

    # Styling
    ax.set_title(doctitles[ansatz], fontsize=14)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.grid(alpha=0.2)
    if ansatz == "ansatz1":
        ax.set_ylabel(r"$x\,t_{3}(x)$", fontsize=12)
    ax.legend(fontsize=9)

plt.suptitle(
    "Comparison of $t_3(x)$: Fixed-PDF analysis vs Simultaneous fit ($C_{\\rm true}=0$)",
    fontsize=16,
    y=0.98,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(image_dir / "comparison_nonbsm_vs_bsm.png", bbox_inches="tight")
plt.show()


# %%
# alpha vs. beta — SINGLE PLOT WITH UNCERTAINTY ELLIPSES


# Helper to plot a 1sigma confidence ellipse
def plot_confidence_ellipse(ax, data, n_std=1.0, **kwargs):  # noqa: ANN001, ANN003, ANN201
    """Plot an n_std confidence ellipse of `data` (2xN array) on `ax`.

    Returns the ellipse center (mean_x, mean_y).
    """
    x, y = data
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse((mean_x, mean_y), width, height, angle=theta, **kwargs)
    ax.add_patch(ellipse)
    return mean_x, mean_y


# Single axes
fig, ax = plt.subplots(figsize=(8, 6))

# 1) No BSM fits: plot ellipses without legend, crosses with legend
left_configs = {
    "fit_real_real": "Real-Data Fit",
    "fit_pseudo_replica": "Pseudo-Replica Fit",
}
color_map_left = {"fit_real_real": "C3", "fit_pseudo_replica": "C4"}
for prefix, label in left_configs.items():
    subset = df_results[df_results["config_key"].str.startswith(prefix)]
    if subset.empty:
        continue
    alphas = subset["alpha"].to_numpy()
    betas = subset["beta"].to_numpy()
    # ellipse, no label
    mx, my = plot_confidence_ellipse(
        ax,
        (alphas, betas),
        edgecolor=color_map_left[prefix],
        facecolor="none",
        linewidth=2,
        label=None,
    )
    # mark mean with cross, labeled
    ax.scatter(mx, my, marker="x", color=color_map_left[prefix], s=50, label=label)

# 2) BSM sensitivity scans: ellipses colored by C_true, markers by ansatz
marker_map = {"ansatz1": "s", "ansatz2": "o"}
for ansatz in ansatzes:
    for C in C_trues:
        cfg_key = f"sens_{ansatz}_C{C:.0e}"
        subset = df_results[df_results["config_key"] == cfg_key]
        if subset.empty:
            continue
        alphas = subset["alpha"].to_numpy()
        betas = subset["beta"].to_numpy()
        # ellipse, no label
        mx, my = plot_confidence_ellipse(
            ax,
            (alphas, betas),
            edgecolor=color_map[C],
            facecolor="none",
            linewidth=2,
            label=None,
        )
        # mean marker, no label
        ax.scatter(
            mx,
            my,
            marker=marker_map[ansatz],
            color=color_map[C],
            edgecolor="k",
            s=60,
            label=None,
        )

# Build combined legend under the plot
# 1) configuration (crosses)
config_handles = []
for prefix, label in left_configs.items():
    config_handles.append(
        Line2D(
            [0],
            [0],
            marker="x",
            color=color_map_left[prefix],
            linestyle="None",
            markersize=8,
            label=label,
        ),
    )
# 2) ansatz markers
ansatz_handles = [
    Line2D(
        [0],
        [0],
        marker=marker_map[a],
        color="gray",
        linestyle="None",
        markeredgecolor="k",
        markersize=8,
        label=f"Ansatz {a[-1]}",
    )
    for a in ansatzes
]
# 3) C_true colors
ct_handles = [Line2D([0], [0], color=color_map[C], lw=4, label=f"{C:g}") for C in C_trues]

# Combine and place legend
all_handles = config_handles + ansatz_handles + ct_handles
all_labels = [h.get_label() for h in all_handles]
fig.legend(
    handles=all_handles,
    labels=all_labels,
    ncol=len(all_handles),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    fontsize=9,
    frameon=False,
)

ax.set_xlabel(r"$\alpha$", fontsize=12)
ax.set_ylabel(r"$\beta$", fontsize=12)
ax.set_title("Alpha vs. Beta: Uncertainty Ellipses", fontsize=14)
ax.grid(alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(image_dir / "alpha_beta_comp.png", bbox_inches="tight")
plt.show()

# %%
# alpha & beta distributions by ansatz and C_true (updated)


# 2 rows x 2 cols: rows=parameters (alpha, beta), cols=ansatzes
grid_kw = {"wspace": 0, "hspace": 0.3}
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharey=True, gridspec_kw=grid_kw)

for j, ansatz in enumerate(ansatzes):
    axes[0, j].set_title(doctitles[ansatz], fontsize=14)
    for i, param in enumerate(["alpha", "beta"]):
        ax = axes[i, j]
        for C in C_trues:
            cfg = f"sens_{ansatz}_C{C:.0e}"
            df_sub = df_results[df_results["config_key"] == cfg]
            if df_sub.empty:
                continue
            vals = df_sub[param].to_numpy()
            mean_val = vals.mean()
            std_val = vals.std()
            # histogram
            ax.hist(vals, bins=20, density=True, alpha=0.3, color=color_map[C])
            # vertical mean line
            ax.axvline(
                mean_val,
                color=color_map[C],
                linestyle="--",
                linewidth=2,
                label=rf"$\mu={mean_val:.3f},\ \sigma={std_val:.3f}$",
            )
        # labels
        ax.set_xlabel(r"$\alpha$" if param == "alpha" else r"$\beta$", fontsize=12)
        if j == 0:
            ax.set_ylabel("Density", fontsize=12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

# Create global legend for C_true colours
ct_handles = [
    Line2D([0], [0], color=color_map[C], lw=4, label=f"$C_{{\\rm true}}={C:g}$") for C in C_trues
]
fig.legend(
    handles=ct_handles,
    ncol=len(C_trues),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    fontsize=10,
    frameon=False,
)

# Adjust layout to make room for the legend at the bottom
plt.savefig(image_dir / "alpha_beta_histograms_with_ct_legend.png", bbox_inches="tight")
plt.show()

# %%
#  Raw C_fit Histograms
fig, axes = plt.subplots(
    nrows=1,
    ncols=len(C_trues),
    figsize=(4 * len(C_trues), 4),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0},  # no gap between panels
)

ansatz_colors = {"ansatz1": "orange", "ansatz2": "C0"}

for col_idx, C_true_val in enumerate(C_trues):
    ax = axes[col_idx]

    for i, ansatz_name in enumerate(["ansatz1", "ansatz2"]):
        cfg = f"sens_{ansatz_name}_C{C_true_val:.0e}"
        df_sub = df_results[df_results["config_key"] == cfg]
        if df_sub.empty:
            continue

        vals = df_sub["C_fit"].to_numpy()
        mu, sigma = vals.mean(), vals.std()

        # filled histogram
        ax.hist(
            vals,
            bins=30,
            histtype="stepfilled",
            alpha=0.5,
            density=True,
            color=ansatz_colors[ansatz_name],
        )
        # dashed mean line, colour-matched per ansatz
        ax.axvline(
            mu,
            color=ansatz_colors[ansatz_name],
            linestyle="--",
            linewidth=1.5,
        )

        # coloured mu/sigma text
        ax.text(
            0.95,
            0.95 - i * 0.07,
            rf"$\mu={mu:.3f},\ \sigma={sigma:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=ansatz_colors[ansatz_name],
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
        )

    # true-value line remains black
    ax.axvline(C_true_val, color="k", linestyle="-", lw=1)

    ax.set_title(rf"$C_{{\rm true}} = {C_true_val:g}$", fontsize=12)
    ax.set_xlabel(r"$C_{\rm fit}$", fontsize=12)
    if col_idx == 0:
        ax.set_ylabel("Density", fontsize=12)
    ax.grid(alpha=0.2)

# global legend: ansatz colours + black line for true C
legend_handles = [
    Line2D([0], [0], color=ansatz_colors["ansatz1"], lw=4, label="Ansatz 1"),
    Line2D([0], [0], color=ansatz_colors["ansatz2"], lw=4, label="Ansatz 2"),
    Line2D([0], [0], color="k", lw=1, label="True $C$"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.01),
    frameon=False,
    fontsize=10,
)

plt.suptitle(
    "$C_{\\rm fit}$ Distributions",
    y=1,
    fontsize=14,
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(image_dir / "histograms.png", bbox_inches="tight")
plt.show()

# %%
# C-distribution, PDF-fixed vs. Joint-fit for both ansatz1 and ansatz2

# --- build fixed-PDF ensemble from fit_pseudo_replica ---
f_list = df_results.loc[df_results["config_key"] == "fit_pseudo_replica", "f_raw_best"].to_numpy()
f_stack = np.vstack([arr.cpu().numpy() if hasattr(arr, "cpu") else arr for arr in f_list])
f_fixed = f_stack.mean(axis=0)

# --- ansatz vectors ---
K1_raw = (q2_vals - q2_vals.min()) ** 2
K1 = K1_raw / np.max(np.abs(K1_raw))

K2_raw = x_vals_data * (1.0 - x_vals_data) * (q2_vals - q2_vals.min())
K2 = K2_raw / np.max(np.abs(K2_raw))

# --- precompute A and B ---
A = W.dot(f_fixed)
B1 = A * K1
B2 = A * K2

Cinv = np.linalg.inv(c_yy)
n_replicas = 100

# --- analytic fit of C for each replica ---
C_fixed1, C_fixed2 = [], []
for replica in range(n_replicas):
    rng = np.random.default_rng(seed=replica * 451)
    y_i = rng.multivariate_normal(y_theory, c_yy)
    num1 = B1.dot(Cinv.dot(y_i - A))
    den1 = B1.dot(Cinv.dot(B1))
    C_fixed1.append(num1 / den1)
    num2 = B2.dot(Cinv.dot(y_i - A))
    den2 = B2.dot(Cinv.dot(B2))
    C_fixed2.append(num2 / den2)

C_fixed1 = np.array(C_fixed1)
C_fixed2 = np.array(C_fixed2)

# --- extract joint-fit C from network runs ---
C_joint1 = df_results.loc[df_results["config_key"] == "sens_ansatz1_C0e+00", "C_fit"].to_numpy()
C_joint2 = df_results.loc[df_results["config_key"] == "sens_ansatz2_C0e+00", "C_fit"].to_numpy()

# --- plotting distributions ---
fig, axes = plt.subplots(ncols=2, figsize=(12, 5), sharey=True, gridspec_kw={"wspace": 0})
fig.subplots_adjust(bottom=0.2)

for ax, C_fixed, C_joint, title in zip(
    axes,
    [C_fixed1, C_fixed2],
    [C_joint1, C_joint2],
    ["Ansatz 1", "Ansatz 2"],
):
    # common bins
    all_C = np.concatenate([C_fixed, C_joint])  # noqa: N816
    bins = np.linspace(all_C.min(), all_C.max(), 30)
    # histograms
    h1 = ax.hist(C_fixed, bins=bins, density=True, alpha=0.6, color="C0")
    h2 = ax.hist(C_joint, bins=bins, density=True, alpha=0.6, color="C1")
    # mean and sigma
    mu_fixed, sigma_fixed = np.mean(C_fixed), np.std(C_fixed)
    mu_joint, sigma_joint = np.mean(C_joint), np.std(C_joint)
    ax.axvline(
        mu_fixed,
        linestyle="--",
        color="red",
        label=rf"$\mu_{{fix}}={mu_fixed:.3f}\pm{sigma_fixed:.3f}$",
    )
    ax.axvline(
        mu_joint,
        linestyle="--",
        color="blue",
        label=rf"$\mu_{{simu}}={mu_joint:.3f}\pm{sigma_joint:.3f}$",
    )
    # true value
    ax.axvline(0, linestyle=":", color="black", label=r"$C_{true}=0$")
    ax.set_xlabel(r"$C_{\mathrm{fit}}$")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

# global legend for histogram colors
legend_elements = [
    Patch(facecolor="C0", alpha=0.6, label="Fixed-PDF"),
    Patch(facecolor="C1", alpha=0.6, label="Simultaneous-Fit"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2)

axes[0].set_ylabel("Density")
fig.suptitle(
    "Distribution of fitted Wilson coefficient (closure test, "
    "$C_{\\rm true}=0$): Fixed-PDF analysis vs Simultaneous fit",
    y=0.97,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(image_dir / "moneyplot2_C_distribution_fixed.png", bbox_inches="tight")
plt.show()


# %%
