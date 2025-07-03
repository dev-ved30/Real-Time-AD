#!/bin/bash
#SBATCH --account=b1094
#SBATCH --partition=ciera-std
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=52
#SBATCH --mem=90G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vedshah2029@u.northwestern.edu

cd /projects/b1094/rehemtulla/BTSbot
module purge all
conda deactivate
source activate BTSbotv2
conda env list

python train_torch.py train_configs/v2_config.json
