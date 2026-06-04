#!/bin/bash
## (1) Set a name for the job
#$ -N run_traj
# (2) Required resourses
#$ -pe smp 1
# (3) Output files
#$ -cwd
#$ -q cerqt03.q
#$ -o prueba.out
#$ -e prueba.err
# (4) Send mail at the end of the run.
#$ -m e
#$ -M YOUR-EMAIL@gmail.com
#$ -S /bin/bash
#
. /etc/profile
module load anaconda/2024.10
source activate vmd_env

START=$(date +%s)

python new_Jordi_trajectory_to_xyz.py  > traj2xyz.log

END=$(date +%s)
DIFF=$(( $END - $START ))
echo temps: $DIFF

