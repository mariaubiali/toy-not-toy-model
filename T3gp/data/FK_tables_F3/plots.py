import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# Choose file here
# ------------------------------------------------------------

# DATAFILE = "Dataset/data_NC_208.npz"
# DATAFILE = "Dataset/data_CC_ep_208.npz"
DATAFILE = "Dataset/data_CC_em_208.npz"


def get_cov_key(data):
    if "c_yy" in data.files:
        return "c_yy"
    if "cov" in data.files:
        return "cov"
    raise KeyError("Could not find covariance key. Expected 'c_yy' or 'cov'.")


def main():
    data = np.load(DATAFILE, allow_pickle=True)

    print("Loaded:", DATAFILE)
    print("Keys:", data.files)
    print()

    # ------------------------------------------------------------
    # Core variables
    # ------------------------------------------------------------
    q2_vals = data["q2_vals"]
    x_vals = data["x_vals"]
    y_vals = data["y_vals"]
    y_pseudo = data["y_pseudo"]
    W = data["W"]
    xgrid = data["xgrid"]
    y_theory = data["y_theory"]
    xtarget_true = data["xt3_true"]
    kinematics = data["kinematics"]

    cov_key = get_cov_key(data)
    cov = data[cov_key]

    y_direct = data["y_direct"] if "y_direct" in data.files else None
    y_kin_vals = data["y_kin_vals"] if "y_kin_vals" in data.files else None
    Q0 = data["Q0"][0] if "Q0" in data.files else None

    channel_codes = data["channel_codes"] if "channel_codes" in data.files else None
    channel_names = data["channel_names"] if "channel_names" in data.files else None

    xtarget_true_full = data["xtarget_true_full"] if "xtarget_true_full" in data.files else None
    xtarget_true_full_2d = data["xtarget_true_full_2d"] if "xtarget_true_full_2d" in data.files else None
    W_tensor = data["W_tensor"] if "W_tensor" in data.files else None

    uv_true = data["uv_true"] if "uv_true" in data.files else None
    dv_true = data["dv_true"] if "dv_true" in data.files else None
    sv_true = data["sv_true"] if "sv_true" in data.files else None
    cv_true = data["cv_true"] if "cv_true" in data.files else None

    print("Shapes:")
    for k in data.files:
        try:
            print(f"{k:20s} {data[k].shape}")
        except Exception:
            print(f"{k:20s} scalar")
    print()

    # ------------------------------------------------------------
    # 1. Kinematic coverage
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        kinematics[:, 0],
        kinematics[:, 1],
        c=y_vals,
        s=18,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel(r"$Q^2$")
    plt.title("Kinematic coverage coloured by y_vals")
    plt.colorbar(sc, label="y_vals")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        kinematics[:, 0],
        kinematics[:, 1],
        c=y_pseudo,
        s=18,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel(r"$Q^2$")
    plt.title("Kinematic coverage coloured by y_pseudo")
    plt.colorbar(sc, label="y_pseudo")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(
        kinematics[:, 0],
        kinematics[:, 1],
        c=y_theory,
        s=18,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel(r"$Q^2$")
    plt.title("Kinematic coverage coloured by y_theory")
    plt.colorbar(sc, label="y_theory")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 2. Observable vectors
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(y_vals, ".", label="y_vals", alpha=0.8)
    plt.plot(y_pseudo, ".", label="y_pseudo", alpha=0.7)
    plt.plot(y_theory, ".", label="y_theory", alpha=0.8)
    if y_direct is not None:
        plt.plot(y_direct, ".", label="y_direct", alpha=0.6)
    plt.xlabel("Data index")
    plt.ylabel("Observable")
    plt.title("Observable vectors")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 3. Covariance
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(np.diag(cov), label=f"diag({cov_key})")
    plt.xlabel("Index")
    plt.ylabel("Variance")
    plt.title("Covariance diagonal")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.imshow(cov, aspect="auto", origin="lower")
    plt.title(f"Covariance matrix: {cov_key}")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 4. FK operator
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.imshow(W, aspect="auto", origin="lower")
    plt.title("Reduced operator W")
    plt.xlabel("x-grid index")
    plt.ylabel("Data index")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    if W_tensor is not None:
        plt.figure(figsize=(8, 5))
        plt.imshow(W_tensor.reshape(W_tensor.shape[0], -1), aspect="auto", origin="lower")
        plt.title("Full operator W_tensor reshaped")
        plt.xlabel("Flattened (channel, x) index")
        plt.ylabel("Data index")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # 5. xgrid and true target
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(xgrid)), xgrid, marker="o", ms=3, label="xgrid")
    plt.xlabel("x-grid index")
    plt.ylabel("x")
    plt.title("xgrid")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(xgrid, xtarget_true, marker="o", ms=3, label="xtarget_true")
    plt.xlabel("x")
    plt.ylabel("xtarget_true")
    plt.title("xtarget_true vs xgrid")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(xgrid, xtarget_true, marker="o", ms=3, label="xtarget_true")
    plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel("xtarget_true")
    plt.title("xtarget_true vs xgrid (log x)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # requested explicitly: xgrid vs xtrue plot
    plt.figure(figsize=(7, 4))
    plt.plot(xgrid, xtarget_true, "o-", label="xtrue on xgrid")
    plt.xscale("log")
    plt.xlabel("xgrid")
    plt.ylabel("xtrue / xtarget_true")
    plt.title("xgrid vs xtrue")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 6. Extra true components if available
    # ------------------------------------------------------------
    if uv_true is not None and dv_true is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(xgrid, uv_true, label="uv_true")
        plt.plot(xgrid, dv_true, label="dv_true")
        if sv_true is not None:
            plt.plot(xgrid, sv_true, label="sv_true")
        if cv_true is not None:
            plt.plot(xgrid, cv_true, label="cv_true")
        plt.xscale("log")
        plt.xlabel("x")
        plt.ylabel("Valence component")
        plt.title("Valence components on xgrid")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if xtarget_true_full_2d is not None and channel_codes is not None:
        plt.figure(figsize=(8, 5))
        for i, code in enumerate(channel_codes):
            label = f"code {code}"
            if channel_names is not None and i < len(channel_names):
                label = str(channel_names[i])
            plt.plot(xgrid, xtarget_true_full_2d[i], label=label)
        plt.xscale("log")
        plt.xlabel("x")
        plt.ylabel("Full-basis target")
        plt.title("Full basis components on xgrid")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------
    # 7. Convolution checks
    # ------------------------------------------------------------
    y_pred = W @ xtarget_true

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, y_vals, s=18, alpha=0.7, label="y_pred vs y_vals")
    mn = min(y_pred.min(), y_vals.min())
    mx = max(y_pred.max(), y_vals.max())
    plt.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
    plt.xlabel("y_pred = W @ xtarget_true")
    plt.ylabel("y_vals")
    plt.title("Convolution vs data")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, y_pseudo, s=18, alpha=0.7, label="y_pred vs y_pseudo")
    mn = min(y_pred.min(), y_pseudo.min())
    mx = max(y_pred.max(), y_pseudo.max())
    plt.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
    plt.xlabel("y_pred = W @ xtarget_true")
    plt.ylabel("y_pseudo")
    plt.title("Convolution vs pseudo-data")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, y_theory, s=18, alpha=0.7, label="y_pred vs y_theory")
    mn = min(y_pred.min(), y_theory.min())
    mx = max(y_pred.max(), y_theory.max())
    plt.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
    plt.xlabel("y_pred = W @ xtarget_true")
    plt.ylabel("y_theory")
    plt.title("Convolution vs theory")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 8. Residuals
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(y_vals - y_pred, ".", label="y_vals - y_pred", alpha=0.8)
    plt.plot(y_pseudo - y_pred, ".", label="y_pseudo - y_pred", alpha=0.8)
    plt.plot(y_theory - y_pred, ".", label="y_theory - y_pred", alpha=0.8)
    if y_direct is not None:
        plt.plot(y_direct - y_pred, ".", label="y_direct - y_pred", alpha=0.8)
    plt.xlabel("Data index")
    plt.ylabel("Residual")
    plt.title("Residuals relative to W @ xtarget_true")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 9. Q^2 slices
    # ------------------------------------------------------------
    unique_q2 = np.unique(q2_vals)
    print("Unique Q2 values:", unique_q2)

    nshow = min(4, len(unique_q2))
    chosen_q2 = unique_q2[:nshow]

    plt.figure(figsize=(8, 5))
    for q2 in chosen_q2:
        mask = np.isclose(q2_vals, q2)
        order = np.argsort(x_vals[mask])
        plt.plot(
            x_vals[mask][order],
            y_theory[mask][order],
            "o-",
            label=rf"$Q^2={q2:g}$"
        )
    plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel("y_theory")
    plt.title("Theory curves at selected Q²")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    for q2 in chosen_q2:
        mask = np.isclose(q2_vals, q2)
        order = np.argsort(x_vals[mask])
        plt.plot(
            x_vals[mask][order],
            y_vals[mask][order],
            "o-",
            label=rf"data, $Q^2={q2:g}$"
        )
    plt.xscale("log")
    plt.xlabel("x")
    plt.ylabel("y_vals")
    plt.title("Data curves at selected Q²")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 10. Optional metadata
    # ------------------------------------------------------------
    if Q0 is not None:
        print("Q0 =", Q0)

    if channel_codes is not None:
        print("channel_codes =", channel_codes)

    if "coeff_u" in data.files:
        print("coeff_u =", data["coeff_u"][0])
    if "coeff_d" in data.files:
        print("coeff_d =", data["coeff_d"][0])
    if "coeff_s" in data.files:
        print("coeff_s =", data["coeff_s"][0])
    if "coeff_c" in data.files:
        print("coeff_c =", data["coeff_c"][0])

    if "reco_max_abs_diff" in data.files:
        print("reco_max_abs_diff =", data["reco_max_abs_diff"][0])
    if "reco_rel_max_diff" in data.files:
        print("reco_rel_max_diff =", data["reco_rel_max_diff"][0])


if __name__ == "__main__":
    main()