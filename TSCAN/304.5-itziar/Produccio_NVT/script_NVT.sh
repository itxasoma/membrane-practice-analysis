#!/bin/bash
## (0) Input files
cp ../Equilibrat_NPT/NPT.restart.coor .
cp ../Equilibrat_NPT/NPT.restart.xsc .
## (1) Name for the job
#$ -N run_namd 
# (2) Required resourses
#$ -pe smp 1
# (3) Output files
#$ -cwd
#$ -q cerqt03.q
#$ -o prueba.out
#$ -e prueba.err
# (4) Send mail at the end of the job
#$ -m e
#$ -M YOUR_EMAIL@gmail.com
#$ -S /bin/bash


. /etc/profile
export OMP_NUM_THREADS=1
ulimit -s unlimited

module load namd/2025-12-04

curr_dir=`pwd`
cp -r * $TMPDIR #cp inputs
cd $TMPDIR
cp $curr_dir/NPT.restart.coor ./
cp $curr_dir/NPT.restart.xsc ./

namd3 simNVT.conf > nvt.log 

mkdir -p $curr_dir/$JOB_ID
cp -r * $curr_dir/$JOB_ID/
cp $curr_dir/$JOB_ID/NVT.dcd $curr_dir/

