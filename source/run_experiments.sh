#!/bin/bash

source ~/miniconda3/bin/activate

conda activate mlp

python train_evaluate_image_classification_system.py --batch_size 32 --is_titan False  --is_pretrain True --learning_rate 0.0001   --num_epochs 200   --experiment_name PreTrainBaseModel0001TH6HU24   --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 1 --dataset_path "/home/s2742733/mlp/MLP_CW/PreTrainMIT3sec.csv"

mkdir -p PostBaseModel0001TH6HU24
mkdir -p PostBaseModel0001TH6HU24/saved_models
mkdir -p PostBaseModel0001TH6HU24/result_outputs

cp -v PreTrainBaseModel0001TH6HU24/saved_models/train_model_199.pth PostBaseModel0001TH6HU24/saved_models/train_model_0.pt
cp -v PreTrainBaseModel0001TH6HU24/result_outputs/summary.csv PostBaseModel0001TH6HU24/result_outputs/summary.csv

python train_evaluate_image_classification_system.py --batch_size 32 --is_titan False  --continue_from_epoch 0 --learning_rate 0.0001   --num_epochs 200   --experiment_name PostBaseModel0001TH6HU24 --use_gpu True --transformer_heads 6 --hidden_units 24 --phi 1 --dataset_path "/home/s2742733/mlp/MLP_CW/MIT3sec.csv"

