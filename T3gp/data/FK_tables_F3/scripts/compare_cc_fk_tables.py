from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import numpy as np

from pineappl.fk_table import FkTable
import lhapdf


# ---------- PDF wrappers ----------

class PDFCallable:
    def __init__(self, pdf):
        self.pdf = pdf

    def __call__(self, pid: int, x: float, q2: float) -> float:
        return float(self.pdf.xfxQ2(pid, x, q2))


class FlavorProjector:
    def __init__(self, base_pdf, allowed_pids: set[int]):
        self.base_pdf = base_pdf
        self.allowed_pids = set(allowed_pids)

    def __call__(self, pid: int, x: float, q2: float) -> float:
        if pid in self.allowed_pids:
            return float(self.base_pdf.xfxQ2(pid, x, q2))
        return 0.0


class LinearComboPDF:
    def __init__(self, base_pdf, coeffs: dict[int, float]):
        self.base_pdf = base_pdf
        self.coeffs = dict(coeffs)

    def __call__(self, pid: int, x: float, q2: float) -> float:
        coeff = self.coeffs.get(pid, 0.0)
        if coeff == 0.0:
            return 0.0
        return float(coeff * self.base_pdf.xfxQ2(pid, x, q2))


# ---------- FK helpers ----------

def convolve_fk(fk: FkTable, pdf_callable) -> np.ndarray:
    return np.asarray(
        fk.convolve(
            pdg_convs=fk.convolutions,
            xfxs=[pdf_callable],
        )
    )


def get_bin_triplets(fk: FkTable) -> list[tuple[float, float, float]]:
    """
    Extract (Q2, x, y) from the embedded yadism runcard stored in FK metadata.
    """
    meta = fk.metadata

    if "runcard" not in meta:
        raise KeyError("FK metadata does not contain 'runcard'")

    runcard = json.loads(meta["runcard"])
    observables = runcard["observables"]

    if len(observables) != 1:
        raise ValueError(f"Expected one observable in runcard, got {list(observables.keys())}")

    obs_name = list(observables.keys())[0]
    pts = observables[obs_name]

    bins = []
    for p in pts:
        bins.append((float(p["Q2"]), float(p["x"]), float(p["y"])))

    if len(bins) != fk.bins():
        raise ValueError(
            f"Bin count mismatch: metadata has {len(bins)} points, FK table has {fk.bins()} bins"
        )

    return bins


def compute_projections(fk: FkTable, pdf) -> dict[str, np.ndarray]:
    return {
        "full": convolve_fk(fk, PDFCallable(pdf)),
        "u": convolve_fk(fk, FlavorProjector(pdf, {2})),
        "ubar": convolve_fk(fk, FlavorProjector(pdf, {-2})),
        "d": convolve_fk(fk, FlavorProjector(pdf, {1})),
        "dbar": convolve_fk(fk, FlavorProjector(pdf, {-1})),
        "u_minus_dbar": convolve_fk(fk, LinearComboPDF(pdf, {2: +1.0, -1: -1.0})),
        "d_minus_ubar": convolve_fk(fk, LinearComboPDF(pdf, {1: +1.0, -2: -1.0})),
    }


def round_key(q2: float, x: float, y: float, ndigits: int = 12):
    return (round(q2, ndigits), round(x, ndigits), round(y, ndigits))


def save_comparison_csv(
    outpath: Path,
    bins_em: list[tuple[float, float, float]],
    bins_ep: list[tuple[float, float, float]],
    em: dict[str, np.ndarray],
    ep: dict[str, np.ndarray],
) -> None:
    em_map = {}
    for i, (q2, x, y) in enumerate(bins_em):
        em_map[round_key(q2, x, y)] = {
            "Q2": q2,
            "x": x,
            "y": y,
            "full": float(em["full"][i]),
            "u": float(em["u"][i]),
            "ubar": float(em["ubar"][i]),
            "d": float(em["d"][i]),
            "dbar": float(em["dbar"][i]),
            "u_minus_dbar": float(em["u_minus_dbar"][i]),
            "d_minus_ubar": float(em["d_minus_ubar"][i]),
        }

    ep_map = {}
    for i, (q2, x, y) in enumerate(bins_ep):
        ep_map[round_key(q2, x, y)] = {
            "Q2": q2,
            "x": x,
            "y": y,
            "full": float(ep["full"][i]),
            "u": float(ep["u"][i]),
            "ubar": float(ep["ubar"][i]),
            "d": float(ep["d"][i]),
            "dbar": float(ep["dbar"][i]),
            "u_minus_dbar": float(ep["u_minus_dbar"][i]),
            "d_minus_ubar": float(ep["d_minus_ubar"][i]),
        }

    all_keys = sorted(
        set(em_map.keys()) | set(ep_map.keys()),
        key=lambda k: (k[0], k[1], k[2]),
    )

    fields = [
        "bin",
        "Q2",
        "x",
        "y",
        "has_em",
        "has_ep",
        "em_full",
        "ep_full",
        "em_u",
        "em_ubar",
        "em_d",
        "em_dbar",
        "ep_u",
        "ep_ubar",
        "ep_d",
        "ep_dbar",
        "em_u_minus_dbar",
        "em_d_minus_ubar",
        "ep_u_minus_dbar",
        "ep_d_minus_ubar",
        "em_minus_ep_full",
        "em_over_ep_full",
    ]

    def fmt(v):
        if v is None:
            return ""
        return f"{v:.16e}"

    with outpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, key in enumerate(all_keys):
            em_row = em_map.get(key)
            ep_row = ep_map.get(key)

            q2 = em_row["Q2"] if em_row is not None else ep_row["Q2"]
            x = em_row["x"] if em_row is not None else ep_row["x"]
            y = em_row["y"] if em_row is not None else ep_row["y"]

            em_full = em_row["full"] if em_row is not None else None
            ep_full = ep_row["full"] if ep_row is not None else None

            writer.writerow(
                {
                    "bin": i,
                    "Q2": fmt(q2),
                    "x": fmt(x),
                    "y": fmt(y),
                    "has_em": em_row is not None,
                    "has_ep": ep_row is not None,
                    "em_full": fmt(em_full),
                    "ep_full": fmt(ep_full),
                    "em_u": fmt(em_row["u"] if em_row is not None else None),
                    "em_ubar": fmt(em_row["ubar"] if em_row is not None else None),
                    "em_d": fmt(em_row["d"] if em_row is not None else None),
                    "em_dbar": fmt(em_row["dbar"] if em_row is not None else None),
                    "ep_u": fmt(ep_row["u"] if ep_row is not None else None),
                    "ep_ubar": fmt(ep_row["ubar"] if ep_row is not None else None),
                    "ep_d": fmt(ep_row["d"] if ep_row is not None else None),
                    "ep_dbar": fmt(ep_row["dbar"] if ep_row is not None else None),
                    "em_u_minus_dbar": fmt(em_row["u_minus_dbar"] if em_row is not None else None),
                    "em_d_minus_ubar": fmt(em_row["d_minus_ubar"] if em_row is not None else None),
                    "ep_u_minus_dbar": fmt(ep_row["u_minus_dbar"] if ep_row is not None else None),
                    "ep_d_minus_ubar": fmt(ep_row["d_minus_ubar"] if ep_row is not None else None),
                    "em_minus_ep_full": fmt(em_full - ep_full) if em_full is not None and ep_full is not None else "",
                    "em_over_ep_full": fmt(em_full / ep_full) if em_full is not None and ep_full not in (None, 0.0) else "",
                }
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--em",
        default="data/fktables/208/HERA_CC_318GEV_EM-SIGMARED.pineappl.lz4",
        help="Electron-beam CC FK table",
    )
    parser.add_argument(
        "--ep",
        default="data/fktables/208/HERA_CC_318GEV_EP-SIGMARED.pineappl.lz4",
        help="Positron-beam CC FK table",
    )
    parser.add_argument(
        "--pdfset",
        default="NNPDF40_nnlo_as_01180",
        help="LHAPDF set name",
    )
    parser.add_argument(
        "--member",
        type=int,
        default=0,
        help="LHAPDF member number",
    )
    parser.add_argument(
        "--outcsv",
        default="fk_cc_em_ep_comparison.csv",
        help="Comparison CSV output",
    )
    parser.add_argument(
        "--outjson",
        default="fk_cc_em_ep_summary.json",
        help="Summary JSON output",
    )
    args = parser.parse_args()

    em_path = Path(args.em)
    ep_path = Path(args.ep)

    print(f"Loading EM FK: {em_path}")
    fk_em = FkTable.read(str(em_path))

    print(f"Loading EP FK: {ep_path}")
    fk_ep = FkTable.read(str(ep_path))

    lhapdf.setVerbosity(0)
    pdf = lhapdf.mkPDF(args.pdfset, args.member)

    print("Convolving EM table...")
    em = compute_projections(fk_em, pdf)

    print("Convolving EP table...")
    ep = compute_projections(fk_ep, pdf)

    bins_em = get_bin_triplets(fk_em)
    bins_ep = get_bin_triplets(fk_ep)

    save_comparison_csv(Path(args.outcsv), bins_em, bins_ep, em, ep)

    summary = {
        "em_file": str(em_path),
        "ep_file": str(ep_path),
        "pdfset": args.pdfset,
        "member": args.member,
        "em_n_bins": fk_em.bins(),
        "ep_n_bins": fk_ep.bins(),
        "em_channels": fk_em.channels(),
        "ep_channels": fk_ep.channels(),
        "first5_em_full": em["full"][:5].tolist(),
        "first5_ep_full": ep["full"][:5].tolist(),
        "first5_em_u_minus_dbar": em["u_minus_dbar"][:5].tolist(),
        "first5_ep_d_minus_ubar": ep["d_minus_ubar"][:5].tolist(),
    }
    Path(args.outjson).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote CSV:  {args.outcsv}")
    print(f"Wrote JSON: {args.outjson}")

    print("\nFirst five bins:")
    print("EM full           :", em["full"][:5])
    print("EP full           :", ep["full"][:5])
    print("EM u-dbar         :", em["u_minus_dbar"][:5])
    print("EM d-ubar         :", em["d_minus_ubar"][:5])
    print("EP u-dbar         :", ep["u_minus_dbar"][:5])
    print("EP d-ubar         :", ep["d_minus_ubar"][:5])


if __name__ == "__main__":
    main()
