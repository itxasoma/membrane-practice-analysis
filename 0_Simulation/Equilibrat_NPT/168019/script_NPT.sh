#!/bin/bash
## (1) Name for the job
#$ -N run_namd 
# (2) Required resourses
#$ -pe smp 1
# (3) Output files
#$ -cwd
#$ -q cerqt03.q
#$ -o prueba.out
#$ -e prueba.err
# (4) Send mail at the end
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
##echo "namd3 lo que sea"
##echo "los ficheros charmm /aplic/charmm_force_field/toppar_c36_jul16"

namd3 simNPT.conf > npt.log 

mkdir -p $curr_dir/$JOB_ID
cp -r * $curr_dir/$JOB_ID/
cp $curr_dir/$JOB_ID/NPT.restart.coor ./
cp $curr_dir/$JOB_ID/NPT.restart.xsc ./

