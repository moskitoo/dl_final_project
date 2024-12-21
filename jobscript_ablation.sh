#!/bin/sh

### -------------specify queue name ----------------
#BSUB -q c02516

### -------------specify GPU request----------------
#BSUB -gpu "num=1:mode=exclusive_process"

### -------------specify job name ----------------
#BSUB -J resnet18_classification_sweep

### -------------specify number of cores ----------------
#BSUB -n 4 
#BSUB -R "span[hosts=1]"

### -------------specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"

### -------------specify wall-clock time (max allowed is 12:00)----------------
#BSUB -W 12:00

#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err

# Activate the virtual environment
source ../dl_cv/bin/activate

# Run the training script
python ablation_channel.py