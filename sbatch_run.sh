#!/bin/bash -l


#SBATCH --account bil-ai --partition tier3
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
module purge
conda activate dplearning

python3 -u train_gan.py --data_path /home/ut3320/learning-with-noisy-labels-benchmark/data/ --dataset $dataset

