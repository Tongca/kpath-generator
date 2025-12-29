
import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import seekpath


# ============================================================
# AFLOW HIGH-SYMMETRY PATH TABLE (STRICT)
# ============================================================

AFLOW_PATHS = {
    "cubic": {
        "main": [
            ("G", "X"), ("X", "W"), ("W", "K"), ("K", "G"),
            ("G", "L"), ("L", "U"), ("U", "W"), ("W", "L"), ("L", "K")
        ],
        "branch": [("U", "X")]
    },
    "tetragonal": {
        "main": [
            ("G", "X"), ("X", "M"), ("M", "G"),
            ("G", "Z"), ("Z", "R"), ("R", "A"), ("A", "Z")
        ],
        "branch": []
    },
    "orthorhombic": {
        "main": [
            ("G", "X"), ("X", "S"), ("S", "Y"), ("Y", "G"),
            ("G", "Z"), ("Z", "U"), ("U", "R"), ("R", "T"), ("T", "Z")
        ],
        "branch": []
    },
    "hexagonal": {
        "main": [
            ("G", "M"), ("M", "K"), ("K", "G"),
            ("G", "A"), ("A", "L"), ("L", "H"), ("H", "A")
        ],
        "branch": []
    },
    "rhombohedral": {
        "main": [
            ("G", "L"), ("L", "B"), ("B", "Z"), ("Z", "G"),
            ("G", "F"), ("F", "L")
        ],
        "branch": []
    },
}


# ============================================================
# UTILS
# ============================================================

def display_label(label: str) -> str:
    return "Γ" if label == "G" else label


def guess_family(structure: Structure) -> str:
    try:
        sga = SpacegroupAnalyzer(structure, symprec=1e-3)
        lattice = sga.get_lattice_type()
    except Exception:
        lattice = None

    mapping = {
        "cubic": "cubic",
        "tetragonal": "tetragonal",
        "orthorhombic": "orthorhombic",
        "hexagonal": "hexagonal",
        "trigonal": "rhombohedral",
        "rhombohedral": "rhombohedral",
    }

    if lattice in mapping:
        return mapping[lattice]

    print("WARNING: Unknown lattice type → fallback to cubic")
    return "cubic"


def normalize_gamma(point_coords: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Ensure Γ exists and is named 'G'
    """
    if "G" in point_coords:
        return point_coords

    for k in list(point_coords.keys()):
        if k.upper().startswith("G"):
            point_coords["G"] = point_coords[k]
            return point_coords

    raise RuntimeError("Γ point not found in SeeK-path output.")


def validate_full_path(path: List[Tuple[str, str]], coords: Dict[str, np.ndarray]):
    missing = set()
    for a, b in path:
        if a not in coords:
            missing.add(a)
        if b not in coords:
            missing.add(b)

    if missing:
        raise RuntimeError(
            "AFLOW path invalid, missing k-points: "
            + ", ".join(sorted(missing))
        )


# ============================================================
# WRITERS
# ============================================================

def write_phonopy(result, segments, npoints):
    labels = [segments[0][0]] + [b for _, b in segments]

    with open("band.conf", "w") as f:
        f.write("BAND =\n")
        for a, b in segments:
            ka = result["point_coords"][a]
            kb = result["point_coords"][b]
            f.write(
                f" {ka[0]:.4f} {ka[1]:.4f} {ka[2]:.4f}"
                f" {kb[0]:.4f} {kb[1]:.4f} {kb[2]:.4f}\n"
            )
        f.write(f"\nBAND_POINTS = {npoints}\n")
        f.write("BAND_LABELS = " +
                " ".join(display_label(l) for l in labels) + "\n")


def write_qe(result, segments, npoints):
    labels = [segments[0][0]] + [b for _, b in segments]

    with open("KPOINTS.qe", "w") as f:
        f.write("K_POINTS crystal_b\n")
        f.write(f"{len(labels)}\n")
        for i, lab in enumerate(labels):
            k = result["point_coords"][lab]
            w = npoints if i < len(labels) - 1 else 1
            f.write(
                f"{k[0]:.4f} {k[1]:.4f} {k[2]:.4f} {w:3d}  ! {display_label(lab)}\n"
            )


def write_alamode(result, segments, npoints):
    with open("kpath.in", "w") as f:
        for a, b in segments:
            ka = result["point_coords"][a]
            kb = result["point_coords"][b]
            f.write(
                f"{a:2s} {ka[0]:.4f} {ka[1]:.4f} {ka[2]:.4f}    "
                f"{b:2s} {kb[0]:.4f} {kb[1]:.4f} {kb[2]:.4f}  {npoints}\n"
            )


def plot_kpath(result, segments):
    labels = [segments[0][0]] + [b for _, b in segments]
    coords = np.array([result["point_coords"][l] for l in labels])

    d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    x = np.concatenate([[0], np.cumsum(d)])

    plt.figure(figsize=(7, 2))
    plt.plot(x, np.zeros_like(x), marker="o")
    plt.yticks([])
    plt.xticks(x, [display_label(l) for l in labels])
    plt.xlabel("k-path")
    plt.tight_layout()
    plt.savefig("kpath.png", dpi=200)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser("AFLOW-compliant k-path generator")
    parser.add_argument("--poscar", required=True)
    parser.add_argument("--npoints", type=int, default=50)
    parser.add_argument("--phonopy", action="store_true")
    parser.add_argument("--qe", action="store_true")
    parser.add_argument("--alamode", action="store_true")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    if not any([args.phonopy, args.qe, args.alamode]):
        args.phonopy = args.qe = args.alamode = True

    structure = Structure.from_file(args.poscar)

    cell = (
        structure.lattice.matrix,
        structure.frac_coords,
        structure.atomic_numbers,
    )

    result = seekpath.get_path(cell, with_time_reversal=True)
    result["point_coords"] = normalize_gamma(result["point_coords"])

    family = guess_family(structure)
    path = AFLOW_PATHS[family]

    validate_full_path(path["main"], result["point_coords"])
    segments = path["main"] + path["branch"]

    if args.phonopy:
        write_phonopy(result, segments, args.npoints)

    if args.qe:
        write_qe(result, segments, args.npoints)

    if args.alamode:
        write_alamode(result, segments, args.npoints)

    if args.plot:
        plot_kpath(result, segments)

    print("AFLOW k-path generated successfully (Γ preserved).")


if __name__ == "__main__":
    main()

