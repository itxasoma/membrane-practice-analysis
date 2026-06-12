import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


u = mda.Universe(
    "../0_Simulation/Equilibrat_NPT/T2905K/estructura_membranaDMPC.psf",
    "../0_Simulation/Produccio_NVT/Vctt/NVT.dcd"
)

lipid = u.select_atoms("resname DMPC")
waters = u.select_atoms("resname TIP3")

# Distance shells in Å: (d_min, d_max, output_file)
SHELLS = [
    (0.0, 3.0,   "trajectory_d0_3.xyz"),
    (3.0, 5.0,   "trajectory_d3_5.xyz"),
    (5.0, 10.0,  "trajectory_d5_10.xyz"),
    (10.0, 15.0, "trajectory_d10_15.xyz"),
]

files = {name: open(name, "w") for _, _, name in SHELLS}

try:
    for ts in u.trajectory:
        # Store whole water residues, not loose atoms
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
                    selected[name].append(res)
                    break

        for _, _, name in SHELLS:
            fout = files[name]
            residues = selected[name]

            # TIP3 water = 3 atoms per residue
            fout.write(f"{3 * len(residues)}\n")
            fout.write(f"Frame {ts.frame}\n")

            for res in residues:
                atoms_by_name = {atom.name: atom for atom in res.atoms}

                # Order for each water's atoms
                ordered_names = ["OH2", "H1", "H2"]
                for atom_name in ordered_names:
                    atom = atoms_by_name[atom_name]
                    x, y, z = atom.position
                    fout.write(
                        f"{res.resid:6d} "
                        f"{atom.name:2s} "
                        f"{x:12.4f} "
                        f"{y:12.4f} "
                        f"{z:12.4f}\n"
                    )
finally:
    for f in files.values():
        f.close()

