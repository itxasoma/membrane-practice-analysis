#!/bin/bash
#SBATCH --job-name=NVT_itziar_3045
#SBATCH --output=nvt.log
#SBATCH --error=nvt.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

# -------------------------------------------------------
# NVT Production — itziar — T = 304.5 K
#
# This Vctt folder must contain:
#   simNVT.conf
#   NPT.restart.coor   (copy from ../../Equilibrat_NPT/)
#   NPT.restart.vel    (copy from ../../Equilibrat_NPT/)
#   NPT.restart.xsc    (copy from ../../Equilibrat_NPT/)
#   coordenades_inicials.pdb
#   estructura_membranaDMPC.psf
#   parametres.prm
#
# Copy the restart files:
#   cp ../../Equilibrat_NPT/NPT.restart.coor .
#   cp ../../Equilibrat_NPT/NPT.restart.vel  .
#   cp ../../Equilibrat_NPT/NPT.restart.xsc  .
#   cp ../../Equilibrat_NPT/coordenades_inicials.pdb .
#   cp ../../Equilibrat_NPT/parametres.prm .
#   cp ../../Equilibrat_NPT/estructura_membranaDMPC.psf .
# -------------------------------------------------------

module load namd/2025  # adjust to your cluster module name

namd3 +p${SLURM_CPUS_PER_TASK} simNVT.conf > nvt.log 2>&1
