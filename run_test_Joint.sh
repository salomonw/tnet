#!/bin/bash -l

#$ -N Joint_Salo
#$ -P noc-lab

#Time limit
#$ -l h_rt=48:00:00

#$ -l mem_per_core=16G

module load gurobi
module load python3 

python3 test_Joint.py
