#!/bin/bash

#SBATCH --job-name=fluid_injection
#SBATCH --output=table1.out
#SBATCH --error=table1.err
#SBATCH --time=48:00:00

module load cuda-11.8.0-gcc-11.2.0-kh2t6kp

# Run your commands here
julia --project fluid_injection1D.jl