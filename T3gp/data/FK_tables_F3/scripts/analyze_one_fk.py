from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np

import pineappl
from pineappl.fk_table import FkTable
import lhapdf


# ---------- PDF wrappers ----------

class PDFCallable:
    """Thin wrapper so we always expose xfx(pid, x, q2)."""

    def __init__(self, pdf):
        self.pdf = pdf

    def __call__(self, pid: int, x: float, q2: float) -> float:
        return float(self.pdf.xfxQ2(pid, x, q2))


class FlavorProjector:
    """
    LHAPDF-like callable that keeps only selected PDG IDs.
    Returns x*f(x,Q2) for allowed flavors, 0 otherwise.
    """

    def __init__(self, base_pdf, allowed_pids: set[int]):
        self.base_pdf = base_pdf
        self.allowed_pids = set(allowed_pids)

    def __call__(self, pid: int, x: float, q2: float) -> float:
        if pid in self.allowed_pids:
            return float(self.base_pdf.xfxQ2(pid, x, q2))
        return 0.0


class LinearComboPDF:
    """
    LHAPDF-like callable representing a linear combination of PDFs:
        sum_i coeff_i * x f_i(x,Q2)

    Example:
        {2: +1}      -> u
        {-1: +1}     -> dbar
        {2: +1, -1: -1} -> u - dbar
    """

    def __init__(self, base_pdf, coeffs: dict[int, float]):
        self.base_pdf = base_pdf
        self.coeffs = dict(coeffs)

    def __call__(self, pid: int, x: float, q2: float) -> float:
        # The FK table asks for one pid at a time, so only return something
        # if that pid is present in the linear combination.
        coeff = self.coeffs.get(pid, 0.0)
        if coeff == 0.0:
            return 0.0
        return float(coeff * self.base_pdf.xfxQ2(pid, x, q2))


# ---------- Helpers ----------

def find_first_fk(fkdir: Path) -> Path:
    matches = sorted(fkdir.glob("*.pineappl.lz4")) + sorted(fkdir.glob("*.pineappl"))
    if not matches:
        raise FileNotFoundError(f"No FK tables found in {fkdir}")
    return matches[0]


def summarize_fk(fk: FkTable) -> dict:
    info = {
        "bins": fk.bins(),
        "bin_dimensions": fk.bin_dimensions(),
        "fac0": fk.fac0(),
        "channels_n": len(fk.channels()),
        "channels": fk.channels(),
        "convolutions": str(fk.convolutions),
        "metadata": fk.metadata,
    }
    return info


def convolve_fk(fk: FkTable, pdf_callable):
    """
    Convolve the FK table with one hadronic PDF.
    For DIS-type FK tables, one convolution object is typically sufficient.
    """
    return np.asarray(
        fk.convolve(
            pdg_convs=fk.convolutions,
            xfxs=[pdf_callable],
        )
    )


def save_results(outpath: Path, results: dict[str, np.ndarray]) -> None:
    keys = list(results.keys())
    n = len(results[keys[0]])
    for k in keys[1:]:
        if len(results[k]) != n:
            raise ValueError(f"Length mismatch for result '{k}'")

    header = ["bin"] + keys
    lines = [",".join(header)]
    for i in range(n):
        row = [str(i)] + [f"{results[k][i]:.16e}" for k in keys]
        lines.append(",".join(row))
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fkdir",
        default="data/fktables/208",
        help="Directory containing FK tables",
    )
    parser.add_argument(
        "--fkfile",
        default=None,
        help="Optional specific FK filename; if omitted, use the first FK in fkdir",
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
        default="fk_one_table_analysis.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--outjson",
        default="fk_one_table_summary.json",
        help="JSON summary output path",
    )
    args = parser.parse_args()

    fkdir = Path(args.fkdir)
    fkpath = Path(args.fkfile) if args.fkfile else find_first_fk(fkdir)

    print(f"Loading FK table: {fkpath}")
    fk = FkTable.read(str(fkpath))

    summary = summarize_fk(fk)
    print("\n=== FK summary ===")
    print(json.dumps(summary, indent=2, default=str))

    # Load PDF
    lhapdf.setVerbosity(0)
    pdf = lhapdf.mkPDF(args.pdfset, args.member)

    # Full prediction
    full_pdf = PDFCallable(pdf)
    pred_full = convolve_fk(fk, full_pdf)

    # Flavor-isolated predictions
    pred_u    = convolve_fk(fk, FlavorProjector(pdf, { 2}))
    pred_ubar = convolve_fk(fk, FlavorProjector(pdf, {-2}))
    pred_d    = convolve_fk(fk, FlavorProjector(pdf, { 1}))
    pred_dbar = convolve_fk(fk, FlavorProjector(pdf, {-1}))

    # CC-style combinations
    pred_u_minus_dbar = convolve_fk(fk, LinearComboPDF(pdf, { 2: +1.0, -1: -1.0}))
    pred_d_minus_ubar = convolve_fk(fk, LinearComboPDF(pdf, { 1: +1.0, -2: -1.0}))

    # Save all results
    results = {
        "full": pred_full,
        "u": pred_u,
        "ubar": pred_ubar,
        "d": pred_d,
        "dbar": pred_dbar,
        "u_minus_dbar": pred_u_minus_dbar,
        "d_minus_ubar": pred_d_minus_ubar,
    }
    save_results(Path(args.outcsv), results)
    Path(args.outjson).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nWrote CSV:  {args.outcsv}")
    print(f"Wrote JSON: {args.outjson}")

    print("\nFirst five bins:")
    for name, arr in results.items():
        print(f"{name:>14}: {arr[:5]}")


if __name__ == "__main__":
    main()
