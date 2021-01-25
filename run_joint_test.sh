#!/bin/bash -l

#$ -l h_rt=24:00:00   # Specify the hard time limit for the job
#$ -pe omp 8
#$ -j y               # Merge the error and output streams into a single file
#$ -P noc-lab

module load python3
module load gurobi
python3 test_Joint.py
