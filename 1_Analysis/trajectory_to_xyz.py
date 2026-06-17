#!/usr/bin/env python3
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


PSF = "../0_Simulation/Equilibrat_NPT/T2905K/estructura_membranaDMPC.psf"
DCD = "../0_Simulation/Produccio_NVT/Vctt/NVT.dcd"


FRAME_STEP = 2  # write every 2nd frame (0.04 ps intervals)


SHELLS = [
    (0.0,  3.0,  "trajectory_d0_3.xyz"),
    (3.0,  5.0,  "trajectory_d3_5.xyz"),
    (5.0,  10.0, "trajectory_d5_10.xyz"),
    (10.0, 15.0, "trajectory_d10_15.xyz"),
]

u = mda.Universe(PSF, DCD)
lipid  = u.select_atoms("resname DMPC")
waters = u.select_atoms("resname TIP3")   # all water atoms (3 per residue)

files = {name: open(name, "w") for _, _, name in SHELLS}

try:
    for ts in u.trajectory[::FRAME_STEP]:

        # One distance_array call over ALL water atoms → shape (n_wat_atoms, n_lip_atoms)
        # min(axis=1) → closest lipid atom for each water atom → shape (n_wat_atoms,)
        # reshape(-1, 3).min(axis=1) → per-residue minimum → shape (n_residues,)
        all_dists = distance_array(
            waters.positions,
            lipid.positions,
            box=u.dimensions
        ).min(axis=1).reshape(-1, 3).min(axis=1)

        for dmin, dmax, name in SHELLS:
            mask = (all_dists > dmin) & (all_dists <= dmax)
            residues = [waters.residues[i] for i in np.where(mask)[0]]

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

