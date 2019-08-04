#!/bin/bash -l

# Set SCC project
#$ -P noc-lab

#$ -pe omp 8

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

module load python3/3.6.5
module load gurobi
python3 main.py