#!/bin/bash
#$ -N run_traj_307.5
#$ -pe smp 1
#$ -cwd
#$ -q cerqt03.q
#$ -o run_traj_307.5.out
#$ -e run_traj_307.5.err
#$ -m e
#$ -M itxaso.mam@gmail.com
#$ -S /bin/bash

. /etc/profile
module load anaconda/2024.10
source activate vmd_env

START=$(date +%s)

python trajectory_to_xyz_tscan.py 307.5 > traj2xyz_307.5.log 2>&1

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "Total time: $DIFF s"
