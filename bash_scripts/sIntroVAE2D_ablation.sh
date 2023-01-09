#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 48:00:00
#SBATCH --cpus-per-gpu 16

# Singularity images can only be used outside /home. In this example the
# image is located here

module load Anaconda/2021.05-nsc1

conda activate /home/x_ioaat/.conda/envs/sIntroVAEv2

for ((i=0; i<16; i++))
do
    echo $i
    python ./scripts/alpha_ablation_vampSIntroVAE.py $i &
done
wait


