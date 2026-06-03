import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
import numpy as np

u = mda.Universe(
    "../0_Simulation/Equilibrat_NPT/estructura_membranaDMPC.psf",
    "../0_Simulation/Produccio_NVT/NVT.dcd"
)

lipid = u.select_atoms("resname DMPC")
waters = u.select_atoms("resname TIP3")

fout = open("trajectory.xyz", "w")

for ts in u.trajectory:

    selected_atoms = []

    # iterate over water residues
    for res in waters.residues:

        d = distance_array(
            res.atoms.positions,
            lipid.positions,
            box=u.dimensions
        )

        min_dist = d.min()

        # equivalent to water3
        if 9.0 < min_dist <= 15.0:
            selected_atoms.extend(res.atoms)

    fout.write(f"{len(selected_atoms)}\n")
    fout.write(f"Frame {ts.frame}\n")

    for atom in selected_atoms:
        x, y, z = atom.position
        fout.write(
            f"{atom.name:2s} "
            f"{x:12.4f} "
            f"{y:12.4f} "
            f"{z:12.4f}\n"
        )

fout.close()

