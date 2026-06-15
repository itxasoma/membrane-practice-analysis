#!/bin/bash
## Input files for corrected NVT production (304.5 K)
cp ../../Equilibrat_NPT/NPT.restart.coor .
cp ../../Equilibrat_NPT/NPT.restart.vel .
cp ../../Equilibrat_NPT/NPT.restart.xsc .
cp ../../Equilibrat_NPT/coordenades_inicials.pdb .
cp ../../Equilibrat_NPT/estructura_membranaDMPC.psf .
cp ../../Equilibrat_NPT/parametres.prm .

#$ -N NVT_itziar_3045
#$ -pe smp 1
#$ -cwd
#$ -q cerqt03.q
#$ -o prueba.out
#$ -e prueba.err
#$ -m e
#$ -M YOUR_EMAIL@gmail.com
#$ -S /bin/bash

. /etc/profile
export OMP_NUM_THREADS=1
ulimit -s unlimited

module load namd/2025-12-04

curr_dir=`pwd`

cp simNVT.conf $TMPDIR/
cp NPT.restart.coor $TMPDIR/
cp NPT.restart.vel $TMPDIR/
cp NPT.restart.xsc $TMPDIR/
cp coordenades_inicials.pdb $TMPDIR/
cp estructura_membranaDMPC.psf $TMPDIR/
cp parametres.prm $TMPDIR/

cd $TMPDIR
namd3 simNVT.conf > nvt.log

mkdir -p $curr_dir/$JOB_ID
cp -r * $curr_dir/$JOB_ID/
cp $curr_dir/$JOB_ID/NVT.dcd $curr_dir/
cp $curr_dir/$JOB_ID/nvt.log $curr_dir/
