#!/bin/bash
#SBATCH --job-name=NVT_sadhbh_2935
#SBATCH --output=nvt.log
#SBATCH --error=nvt.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

# -------------------------------------------------------
# NVT Production — sadhbh — T = 293.5 K
#
# This Vctt folder must contain:
#   simNVT.conf
#   NPT.restart.coor   (symlink or copy from ../run_output/)
#   NPT.restart.vel    (symlink or copy from ../run_output/)
#   NPT.restart.xsc    (symlink or copy from ../run_output/)
#   coordenades_inicials.pdb
#   estructura_membranaDMPC.psf
#   parametres.prm
#
# Copy the restart files from the equilibration run_output:
#   cp ../Equilibrat_NPT/run_output/NPT.restart.coor .
#   cp ../Equilibrat_NPT/run_output/NPT.restart.vel  .
#   cp ../Equilibrat_NPT/run_output/NPT.restart.xsc  .
#   cp ../Equilibrat_NPT/run_output/coordenades_inicials.pdb .
#   cp ../Equilibrat_NPT/run_output/parametres.prm .
#   cp ../../estructura_membranaDMPC.psf . (or from run_output)
# -------------------------------------------------------

module load namd/2025  # adjust to your cluster module name

namd3 +p${SLURM_CPUS_PER_TASK} simNVT.conf > nvt.log 2>&1
