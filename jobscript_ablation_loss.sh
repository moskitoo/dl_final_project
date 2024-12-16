#!/bin/sh

### -------------specify queue name ----------------
#BSUB -q c02516

### -------------specify GPU request----------------
#BSUB -gpu "num=1:mode=exclusive_process"

### -------------specify job name ----------------
#BSUB -J resnet18_classification_sweep

# -- Notify me by email when execution ends   --
#BSUB -N
# -- email address -- 
##BSUB -u s242529@dtu.dk

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
source ~/pytorch_env/bin/activate

# Run the training script
python /zhome/68/f/213210/deep_learning_final_project/dl_final_project/ablation_loss.py