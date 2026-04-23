#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import lhapdf
import matplotlib.pyplot as plt

from validphys.api import API
from validphys.fkparser import load_fktable


# =========================
# CONFIG (edit here only)
# =========================
PDFSET = "NNPDF40_nnlo_as_01180"
MEMBER = 0

SETS = [
    # {
    #     "dataset": "HERA_NC_318GEV_EM-SIGMARED",
    #     "theoryid": 208,
    #     "npz_in": "Dataset/hera_data_nc_em.npz",
    # },
    # {
    #     "dataset": "HERA_CC_318GEV_EP-SIGMARED",
    #     "theoryid": 208,
    #     "npz_in": "Dataset/hera_data_cc_ep.npz",
    # },
    # {
    #     "dataset": "HERA_CC_318GEV_EM-SIGMARED",
    #     "theoryid": 208,
    #     "npz_in": "Dataset/hera_data_cc_em.npz",
    # },
    {
        "dataset": "HERA_NC_225GEV_EP-SIGMARED",
        "theoryid": 208,
        "npz_in": "Dataset/hera_data_nc_ep460.npz",
    },
    {
        "dataset": "HERA_NC_251GEV_EP-SIGMARED",
        "theoryid": 208,
        "npz_in": "Dataset/hera_data_nc_ep575.npz",
    },
    {
        "dataset": "HERA_NC_300GEV_EP-SIGMARED",
        "theoryid": 208,
        "npz_in": "Dataset/hera_data_nc_ep820.npz",
    },
    {
        "dataset": "HERA_NC_318GEV_EP-SIGMARED",
        "theoryid": 208,
        "npz_in": "Dataset/hera_data_nc_ep920.npz",
    },
]

N_X = 400
XMIN = 1e-6
XMAX = 1.0
XJOIN = 1e-1

DO_PLOT = True
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


def Q0_from_dataset_fk(dataset_name, theoryid):
    ds = API.dataset(
        dataset_input={"dataset": dataset_name},
        use_cuts="internal",
        theoryid=theoryid,
    )
    fk = load_fktable(ds.fkspecs[0])
    return float(fk.Q0)


def extend_one(entry):
    npz_in = entry["npz_in"]
    dataset = entry["dataset"]
    theoryid = entry["theoryid"]

    old = np.load(npz_in)
    out = str(Path(npz_in).with_name(Path(npz_in).stem + "_extended.npz"))

    # Q0 from dataset FK
    Q0 = Q0_from_dataset_fk(dataset, theoryid)

    # FK grid reference
    x_fk = np.asarray(old["xgrid"], float)
    xt3_old = np.asarray(old["xt3_true"], float)

    print("xgrid min: ", x_fk.min())
    print("xgrid len: ", len(x_fk))

    # extended grid -> xT3 -> interpolate back
    x_ext = make_xgrid()
    pdf = lhapdf.mkPDF(PDFSET, MEMBER)
    xt3_ext = xT3(pdf, x_ext, Q0)
    xt3_new_fk = np.interp(x_fk, x_ext, xt3_ext)

    # save
    payload = {k: old[k] for k in old.files}
    payload["xgrid_ext"] = x_ext
    payload["xt3_ext"] = xt3_ext
    payload["xt3_interp_on_xgrid"] = xt3_new_fk
    payload["q0"] = np.array([Q0], dtype=float)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)

    print(f"Saved: {out}")
    print(f"Recovered Q0 = {Q0:.6g} GeV")

    # diagnostics
    diff = xt3_new_fk - xt3_old
    print(f"max |Δ| on FK grid: {np.max(np.abs(diff)):.3e}")
    print(
        f"max |Δ/old| on FK grid: "
        f"{np.max(np.abs(diff) / np.maximum(np.abs(xt3_old), 1e-30)):.3e}"
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

        plt.figure()
        plt.plot(x_fk, xt3_old, "o", ms=3, label="old xt3_true (FK)")
        plt.plot(x_ext, xt3_ext, "-", label="new xT3")
        plt.xscale("log")
        plt.xlabel("x")
        plt.ylabel("xT3(x, Q0)")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    for entry in SETS:
        extend_one(entry)


if __name__ == "__main__":
    main()