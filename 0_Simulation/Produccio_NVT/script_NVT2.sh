#!/bin/bash
## (0) Input files only for the corrected equilibration (290.5 K)
cp ../Equilibrat_NPT/T=290.5K/NPT.restart.coor .
cp ../Equilibrat_NPT/T=290.5K/NPT.restart.xsc .

#$ -N run_namd
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

# Copy input files to the temporary directory
cp coordenades_inicials.pdb $TMPDIR/
cp estructura_membranaDMPC.psf $TMPDIR/
cp parametres.prm $TMPDIR/
cp simNVT.conf $TMPDIR/
cp NPT.restart.coor $TMPDIR/
cp NPT.restart.xsc $TMPDIR/

cd $TMPDIR
namd3 simNVT.conf > nvt.log

mkdir -p $curr_dir/$JOB_ID
cp -r * $curr_dir/$JOB_ID/
cp $curr_dir/$JOB_ID/NVT.dcd $curr_dir/
cp $curr_dir/$JOB_ID/nvt.log $curr_dir/   