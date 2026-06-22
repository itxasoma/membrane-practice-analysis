#!/usr/bin/env python3
"""
trajectory_to_xyz_tscan.py — Generate shell-conditioned XYZ trajectories
for all TSCAN temperatures.

For each temperature directory under TSCAN/, this script:
  1. Finds the PSF and NVT.dcd files (handles the different layouts).
  2. Computes per-residue minimum distance to DMPC lipids.
  3. Writes one XYZ file per distance shell into:
       1_Analysis/Vctt-fast-<TEMP_TAG>/trajectory_d*.xyz

Directory layout handled
------------------------
The TSCAN folder has two structural variants:

  Variant A (murphy, JonathanBarrientos):
      TSCAN/<temp>/estructura_membranaDMPC.psf          ← PSF at root
      TSCAN/<temp>/NVT.dcd                              ← DCD at root
      TSCAN/<temp>/Produccio_NVT/<job>/NVT.dcd          ← DCD fallback

  Variant B (sadhbh, Andrea, itziar):
      TSCAN/<temp>/Equilibrat_NPT/estructura_membranaDMPC.psf  ← PSF in NPT
      TSCAN/<temp>/Produccio_NVT/Vctt/<job>/NVT.dcd    ← DCD in job subdir

The script tries all known paths and reports clearly if something is missing.

Usage
-----
    cd <repo-root>
    python 1_Analysis/trajectory_to_xyz_tscan.py             # all temperatures
    python 1_Analysis/trajectory_to_xyz_tscan.py 307.5       # one temperature
    python 1_Analysis/trajectory_to_xyz_tscan.py 301.5 304.5 # subset

Output
------
    1_Analysis/Vctt-fast-<TEMP_TAG>/trajectory_d0_3.xyz
    1_Analysis/Vctt-fast-<TEMP_TAG>/trajectory_d3_5.xyz
    1_Analysis/Vctt-fast-<TEMP_TAG>/trajectory_d5_10.xyz
    1_Analysis/Vctt-fast-<TEMP_TAG>/trajectory_d10_15.xyz
"""

import os
import sys
import glob
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)          # one level up from 1_Analysis
TSCAN_DIR  = os.path.join(REPO_ROOT, 'TSCAN')
ANA_DIR    = SCRIPT_DIR                            # 1_Analysis/

# ─── Settings ────────────────────────────────────────────────────────────────

# Write every FRAME_STEP-th frame.
# DCDfreq=10, timestep=2fs → raw frame = 20 fs = 0.02 ps
# FRAME_STEP=2  → one XYZ frame every 0.04 ps   (fine enough for C_rot)
# FRAME_STEP=10 → one XYZ frame every 0.20 ps   (fast, use for testing)
FRAME_STEP = 2

SHELLS = [
    (0.0,  3.0,  "trajectory_d0_3.xyz"),
    (3.0,  5.0,  "trajectory_d3_5.xyz"),
    (5.0,  10.0, "trajectory_d5_10.xyz"),
    (10.0, 15.0, "trajectory_d10_15.xyz"),
]

# ─── Temperature → directory mapping ─────────────────────────────────────────
# Keys: temperature tag used for output directories and CSV filenames.
# Values: subdirectory name under TSCAN/.

TEMPERATURES = {
    "293.5": "293.5-sadhbh",
    "297.5": "297.5-murphy",
    "301.5": "301.5-Andrea",
    "304.5": "304.5-itziar",
    "307.5": "307.5-JonathanBarrientos",
}

# ─── PSF / DCD finders ───────────────────────────────────────────────────────

def find_psf(temp_dir):
    """Return the path to the PSF file, trying all known locations."""
    candidates = [
        os.path.join(temp_dir, "estructura_membranaDMPC.psf"),
        os.path.join(temp_dir, "Equilibrat_NPT", "estructura_membranaDMPC.psf"),
        os.path.join(temp_dir, "Produccio_NVT", "estructura_membranaDMPC.psf"),
    ]
    candidates += glob.glob(
        os.path.join(temp_dir, "Produccio_NVT", "*", "estructura_membranaDMPC.psf")
    )
    candidates += glob.glob(
        os.path.join(temp_dir, "Produccio_NVT", "Vctt", "*", "estructura_membranaDMPC.psf")
    )
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def find_dcd(temp_dir):
    """Return the path to the NVT DCD file, trying all known locations."""
    candidates = [
        os.path.join(temp_dir, "Produccio_NVT", "NVT.dcd"),
    ]
    # Job subdirectory variants (Vctt/<job>/ and <job>/)
    candidates += glob.glob(
        os.path.join(temp_dir, "Produccio_NVT", "Vctt", "*", "NVT.dcd")
    )
    candidates += glob.glob(
        os.path.join(temp_dir, "Produccio_NVT", "*", "NVT.dcd")
    )
    # Root-level NVT.dcd last (may be a stray test file — lower priority)
    candidates.append(os.path.join(temp_dir, "NVT.dcd"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

# ─── Per-temperature processing ──────────────────────────────────────────────

def process_temperature(temp_tag, subdir):
    print(f"\n{'='*60}")
    print(f"  Temperature : {temp_tag} K   ({subdir})")
    print(f"{'='*60}")

    temp_dir = os.path.join(TSCAN_DIR, subdir)
    if not os.path.isdir(temp_dir):
        print(f"  [ERROR] Directory not found: {temp_dir}")
        print("          (Are you running this on the cluster?)")
        return False

    # Locate PSF
    psf_path = find_psf(temp_dir)
    if psf_path is None:
        print(f"  [ERROR] PSF not found under {temp_dir}")
        print("          Expected: estructura_membranaDMPC.psf")
        return False
    print(f"  PSF  : {os.path.relpath(psf_path, REPO_ROOT)}")

    # Locate DCD
    dcd_path = find_dcd(temp_dir)
    if dcd_path is None:
        print(f"  [ERROR] NVT.dcd not found under {temp_dir}")
        return False
    print(f"  DCD  : {os.path.relpath(dcd_path, REPO_ROOT)}")

    # Output directory
    out_dir = os.path.join(ANA_DIR, f"Vctt-fast-{temp_tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Out  : {os.path.relpath(out_dir, REPO_ROOT)}/")

    # Load universe
    print("  Loading trajectory …")
    u      = mda.Universe(psf_path, dcd_path)
    lipid  = u.select_atoms("resname DMPC")
    waters = u.select_atoms("resname TIP3")

    n_frames_total = len(u.trajectory)
    n_frames_out   = len(range(0, n_frames_total, FRAME_STEP))
    print(f"  Frames in DCD : {n_frames_total}")
    print(f"  Frames to write (step={FRAME_STEP}): {n_frames_out}")
    print(f"  Total time covered : {n_frames_total * 0.02:.1f} ps")

    # Open output files
    shell_paths = {
        name: os.path.join(out_dir, name)
        for _, _, name in SHELLS
    }
    files = {name: open(path, "w") for name, path in shell_paths.items()}

    try:
        for i, ts in enumerate(u.trajectory[::FRAME_STEP]):
            if i % 500 == 0:
                print(f"    frame {i}/{n_frames_out} (DCD frame {ts.frame})", flush=True)

            # Per-residue minimum distance to any lipid atom
            all_dists = (
                distance_array(
                    waters.positions,
                    lipid.positions,
                    box=u.dimensions,
                )
                .min(axis=1)
                .reshape(-1, 3)
                .min(axis=1)
            )

            for dmin, dmax, name in SHELLS:
                mask     = (all_dists > dmin) & (all_dists <= dmax)
                residues = [waters.residues[j] for j in np.where(mask)[0]]

                fout = files[name]
                fout.write(f"{3 * len(residues)}\n")
                fout.write(f"Frame {ts.frame}\n")

                for res in residues:
                    atoms_by_name = {atom.name: atom for atom in res.atoms}
                    for aname in ("OH2", "H1", "H2"):
                        a = atoms_by_name[aname]
                        x, y, z = a.position
                        fout.write(
                            f"{res.resid:6d} "
                            f"{a.name:3s} "
                            f"{x:12.4f} "
                            f"{y:12.4f} "
                            f"{z:12.4f}\n"
                        )
    finally:
        for f in files.values():
            f.close()

    print(f"  Done — files written to {os.path.relpath(out_dir, REPO_ROOT)}/")
    for name, path in shell_paths.items():
        size_mb = os.path.getsize(path) / 1e6 if os.path.isfile(path) else 0
        print(f"    {name:<32s}  {size_mb:6.1f} MB")

    return True

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optionally run only a subset of temperatures:
    #   python trajectory_to_xyz_tscan.py 297.5 304.5
    if len(sys.argv) > 1:
        requested = set(sys.argv[1:])
        temps = {k: v for k, v in TEMPERATURES.items() if k in requested}
        if not temps:
            print(f"No matching temperatures for: {requested}")
            print(f"Available: {list(TEMPERATURES.keys())}")
            sys.exit(1)
    else:
        temps = TEMPERATURES

    print(f"TSCAN root : {TSCAN_DIR}")
    print(f"Output     : 1_Analysis/Vctt-fast-<TEMP_TAG>/")
    print(f"FRAME_STEP : {FRAME_STEP}  (one frame every {FRAME_STEP * 0.02:.3f} ps)")
    print(f"Temperatures to process: {list(temps.keys())}")

    results = {}
    for tag, subdir in temps.items():
        results[tag] = process_temperature(tag, subdir)

    print("\n" + "="*60)
    print("Summary:")
    for tag, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED (check errors above)"
        print(f"  {tag} K  →  {status}")
    print("="*60)
