#!/usr/bin/env python3
import numpy as np
import lhapdf
import matplotlib.pyplot as plt

from validphys.loader import Loader
from validphys.fkparser import load_fktable

# =========================
# CONFIG (edit here only)
# =========================
NPZ_PATH = "Dataset/data_208_L2.npz"
THEORYID = 208
FKSET = "BCDMSP"
PDFSET = "NNPDF40_nnlo_as_01180"
MEMBER = 0

N_X = 400
XMIN = 1e-6
XMAX = 1.0
XJOIN = 1e-1

DO_PLOT = True
OUT_PATH = "Dataset/data_208_L2_extended.npz"
ADD_EXTENDED_TO_NPZ = True
# =========================


def make_xgrid(N=N_X, xmin=XMIN, xmax=XMAX, xjoin=XJOIN):
    x = np.r_[
        np.logspace(np.log10(xmin), np.log10(xjoin), N // 2, endpoint=False),
        np.linspace(xjoin, xmax, N - N // 2),
    ]
    return np.unique(x)


def xT3(pdf, x, Q):
    q = np.full_like(x, Q, dtype=float)
    u = np.asarray(pdf.xfxQ(2, x, q), float)
    ub = np.asarray(pdf.xfxQ(-2, x, q), float)
    d = np.asarray(pdf.xfxQ(1, x, q), float)
    db = np.asarray(pdf.xfxQ(-1, x, q), float)
    return (u + ub) - (d + db)


def Q0_from_fk(theoryid=THEORYID, fkset=FKSET):
    loader = Loader()
    fk = load_fktable(loader.check_fktable(setname=fkset, theoryID=theoryid, cfac=()))
    return float(fk.Q0)


def main():
    old = np.load(NPZ_PATH)
    out = OUT_PATH or NPZ_PATH.replace(".npz", "_extended.npz")

    # q and q2 from dataset
    q2 = old["q2_vals"] if "q2_vals" in old.files else old["kinematics"][:, 1]
    q2 = np.asarray(q2, float)
    q = np.sqrt(q2)

    # Q0 from FK
    Q0 = Q0_from_fk()

    # FK grid reference
    x_fk = np.asarray(old["xgrid"], float)
    xt3_old = np.asarray(old["xt3_true"], float)

    print("xgrid min: ", x_fk.min())
    print("xgrid len: ", len(x_fk))

    # extended grid → xT3 → interpolate back
    x_ext = make_xgrid()
    pdf = lhapdf.mkPDF(PDFSET, MEMBER)
    xt3_ext = xT3(pdf, x_ext, Q0)
    xt3_new_fk = np.interp(x_fk, x_ext, xt3_ext)

    # save
    payload = {k: old[k] for k in old.files}

    # 2) store extended grid + truth as top-level keys
    payload["xgrid_ext"] = x_ext
    payload["xt3_ext"] = xt3_ext

    np.savez(out, **payload)

    print(f"Saved: {out}")
    print(f"Recovered Q0 = {Q0:.6g} GeV")

    # diagnostics
    diff = xt3_new_fk - xt3_old
    print(f"max |Δ| on FK grid: {np.max(np.abs(diff)):.3e}")
    print(
        f"max |Δ/old| on FK grid: {np.max(np.abs(diff) / np.maximum(np.abs(xt3_old), 1e-30)):.3e}"
    )

    if DO_PLOT:
        plt.figure()
        plt.plot(x_fk, xt3_old, "o", ms=3, label="old xt3_true (FK)")
        plt.plot(x_fk, xt3_new_fk, "-", label="new xT3 (interp → FK)")
        plt.xscale("log")
        plt.xlabel("x")
        plt.ylabel("xT3(x, Q0)")
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.plot(x_ext, xt3_ext, "-", label="xT3 (extended grid)")
        plt.xscale("log")
        plt.xlabel("x")
        plt.ylabel("xT3(x, Q0)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(x_fk, xt3_old, "o", ms=3, label="old xt3_true (FK)")
        plt.plot(x_ext, xt3_ext, "-", label="new xT3")
        plt.xscale("log")
        plt.xlabel("x")
        plt.ylabel("xT3(x, Q0)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
