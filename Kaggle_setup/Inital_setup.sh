#!/bin/bash
source ~/miniconda3/bin/activate
conda activate mlp

pip install kaggle

mkdir -p ~/.kaggle
cp /home/s2742733/mlp/MLP_CW/Kaggle_setup/kaggle.json ~/.kaggle/