import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


u = mda.Universe(
    "../0_Simulation/Equilibrat_NPT/167834/estructura_membranaDMPC.psf",
    "../0_Simulation/Produccio_NVT/original/NVT.dcd"
)

lipid = u.select_atoms("resname DMPC")
waters = u.select_atoms("resname TIP3")

# Distance shells in Å: (d_min, d_max, output_file)
SHELLS = [
    (0.0, 3.0,  "trajectory_d0_3.xyz"),
    (3.0, 5.0,  "trajectory_d3_5.xyz"),
    (5.0, 10.0, "trajectory_d5_10.xyz"),
    (10.0, 15.0, "trajectory_d10_15.xyz"),
]

files = {name: open(name, "w") for _, _, name in SHELLS}

for ts in u.trajectory:
    selected = {name: [] for _, _, name in SHELLS}

    for res in waters.residues:
        d = distance_array(
            res.atoms.positions,
            lipid.positions,
            box=u.dimensions
        )
        min_dist = d.min()

        for dmin, dmax, name in SHELLS:
            if dmin < min_dist <= dmax:
                selected[name].extend(res.atoms)

    for _, _, name in SHELLS:
        fout = files[name]
        atoms = selected[name]

        fout.write(f"{len(atoms)}\n")
        fout.write(f"Frame {ts.frame}\n")

        for atom in atoms:
            x, y, z = atom.position
            fout.write(
                f"{atom.name:2s} "
                f"{x:12.4f} "
                f"{y:12.4f} "
                f"{z:12.4f}\n"
            )

for f in files.values():
    f.close()

