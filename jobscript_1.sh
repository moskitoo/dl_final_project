#!/bin/sh
### -------------specify queue name ----------------
#BSUB -q c02516
### -------------specify GPU request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
### -------------specify job name ----------------
#BSUB -J train_unet_model

# -- Notify me by email when execution ends   --
#BSUB -N
# -- email address -- 
##BSUB -u s242529@dtu.dk
### -------------specify number of cores ----------------
#BSUB -n 4 
#BSUB -R "span[hosts=1]"
### -------------specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=32GB]"
### -------------specify wall-clock time (max allowed is 12:00)----------------
#BSUB -W 12:00
#BSUB -o /zhome/68/f/213210/deep_learning_final_project/output_logs/%J.out
#BSUB -e /zhome/68/f/213210/deep_learning_final_project/output_logs/%J.err

# Activate the PyTorch virtual environment
source ~/pytorch_env/bin/activate  # Adjust the virtual environment path if necessary

# Run the training script
python3 /zhome/68/f/213210/deep_learning_final_project/dl_final_project/train.py
