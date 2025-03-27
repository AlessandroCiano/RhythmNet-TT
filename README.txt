# RhythmNet-TT: Adaptive Arrhythmia Detection Beyond Training

This repository contains the code for RhythmNet-TT, a Transformer-based deep learning model for ECG arrhythmia detection.

## Description

RhythmNet-TT is a model that builds upon the ECG-DETR architecture [1]. By incorporating a memory module inspired by the Memory as Context (MAC) framework [2]. This adaptation aims to improve the model's ability to handle variability in ECG signals and process longer ECG sequences.

## Code Structure

The repository is organized as follows:

* 'source/': Contains all code of models ('model_acrchitectures.py', 'titan.py'), experiment training ('arg_extractor.py', 'experiment_builder.py', 'storage_utils.py'), utilities ('loss_functions.py', 'VisualiseResults.ipynb'), along with debugging files.

* 'Kaggle_setup/': instructions and necessary files to obtain MIT-BIH Atrial Fibrillation dataset in our format (3 s, 7 s, 15 s).

* './': Pre-processing code for MIT-BIH Arrhythmia dataset ('Pre-preprocessing_MIT-BIH_Arrhythmia.py'), which requires csv files processed from 'get_csv_MIT-BIH_Arrhithmia.ipynb' 


## Dependencies

The code requires the following packages (python=3.12.5):
[numpy, scipy, matplotlib, jupyter, pytorch, torchvision, torchaudio]


```bash
pip install -r requirements.txt


## References

1.  Hu, Rui, Jie Chen, and Li Zhou. "A transformer-based deep neural network for arrhythmia detection using continuous ECG signals." *Computers in Biology and Medicine* 144 (2022): 105325. https://doi.org/10.1016/j.compbiomed.2022.105325
2.  Behrouz, Ali, Peilin Zhong, and Vahab Mirrokni. "Titans: Learning to memorize at test time." *arXiv preprint arXiv:2501.00663* (2024). https://arxiv.org/abs/2501.00663




