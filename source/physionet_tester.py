import pandas as pd
import numpy as np
from model_architectures import Quite_Big_Titan_Model,Quite_Big_Model
import argparse
import torch
import tqdm

#python physionet_tester.py --MLP_path "C:/Users/User/Desktop/Python Code/Uni/SEM 2/MLP" --experiment_name PostBaseModel0001TH6HU24 --is_titan False --timespan 3 --transformer_heads 6 --hidden_units 24

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#First, get info
def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    """
    parser = argparse.ArgumentParser(
        description='Helper script for training and evaluating models on the ECG dataset.'
    )

    # Model and dataset parameters
    parser.add_argument('--experiment_name', nargs="?", type=str, default="Titan", help='Name of the experiment')
    parser.add_argument('--epoch',nargs="?",type=int,default=-1,help="The epoch to test, -1 if you want to load the last epoch")
    parser.add_argument('--MLP_path', nargs="?", type=str, default="C:/Users/User/Desktop/Python Code/Uni/SEM 2", help='File path of where the MLP file is')

    parser.add_argument('--is_titan', nargs="?", type=str2bool, default=True, help='True if this is the titan model')
    parser.add_argument('--timespan',type=int,default=3,help="The number of seconds the model is for")

    #We need all of the hyperparameters that we used in the model creation
    parser.add_argument('--transformer_heads', nargs="?", type=int, default=6, help='Number of transformer heads')
    parser.add_argument('--hidden_units', nargs="?", type=int, default=24, help='Number of hidden units in final layer')
    parser.add_argument('--phi', nargs="?", type=float, default=1, help='Scale of CNN Backbone')
    parser.add_argument('--alpha', nargs="?", type=float, default=0.1, help='Alpha for neural memory')
    parser.add_argument('--nm_hu', nargs="?", type=int, default=128, help='Neural Memory hidden units')
    parser.add_argument('--nm_kqv_size', nargs="?", type=int, default=64, help='Size of query/key/value in neural memory')
    parser.add_argument('--pers_dim', nargs="?", type=int, default=32, help='Size of the persistent memory')
    parser.add_argument('--nu', nargs="?", type=float, default=0.9, help='Nu for neural memory')
    parser.add_argument('--theta', nargs="?", type=float, default=0.3, help='Theta for neural memory')

    args = parser.parse_args()
    return args

args=get_args()

experiment_name=args.experiment_name
epoch=args.epoch
MLP_path=args.MLP_path
timespan=args.timespan

model_name=f"train_model_{epoch}.pth"
fileName=f"{MLP_path}/MLP_CW/DummyExperiment/pytorch_mlp_framework/{experiment_name}/saved_models/"

if epoch==-1:
    #If the epoch -1, then we need to get the last epoch run
    #Find the final epoch:
    import os #Just to get the file names
    fileNames=os.listdir(fileName)
    for name in fileNames:
        num=int(name[12:-4])
        print(num)
        if num>epoch:
            epoch=num


fileName=f"{MLP_path}/MLP_CW/DummyExperiment/pytorch_mlp_framework/{experiment_name}/saved_models/train_model_{epoch}.pth"

#Time to load the model:

Input_shape=[2,1,timespan*360] #timespan*360Hz... Also, batch size of 2 to make sure the model knows to expect more than 1 batch

if not args.is_titan:
    #If it is the titan model, then set it up
    model=Quite_Big_Model(input_shape=Input_shape,
                                d_model=6,
                                transformer_heads=args.transformer_heads,
                                hidden_units=args.hidden_units,
                                num_classes=5,
                                phi=args.phi)
else:
    model=Quite_Big_Titan_Model(input_shape=Input_shape,
                                d_model=6,
                                transformer_heads=args.transformer_heads,
                                hidden_units=args.hidden_units,
                                num_classes=5,
                                phi=args.phi,
                                nm_hu=args.nm_hu,
                                nm_kqv_size=args.nm_kqv_size,
                                persistent_dim=args.pers_dim,
                                alpha=args.alpha,
                                nu=args.nu,
                                theta=args.theta)



state=torch.load(fileName,weights_only=False,map_location=torch.device('cpu'))
model.load_state_dict(state['network']) #Load the model state


#Now we can have a look at the performace on the physionet stuff
physionetFile=f"{MLP_path}/MLP_CW/physionet_csv_files/ecg_records/"
annotationFile=f"{MLP_path}/MLP_CW/physionet_csv_files/REFERENCE-v3.csv"
annotations=pd.read_csv(annotationFile)
annotations=annotations[annotations["Their_label"]!="~"] #Get rid of the samples that were not classified



#Now we can loop through each annotation and calc some predictions
numCorrect=0

pred_dictionary={
    0:"N",
    1:"A",
    2:"O",#V and F both correspond to the label O
    3:"O"
}

posLabels=np.array(["N","A","O"])#an array of the possible labels

ConfMat=np.zeros((3,4)) #3 rows for the true values, 4 rows for the predictions of the model

for anno_index,recording_num in tqdm.tqdm(enumerate(annotations["Recording"]),total=len(annotations["Recording"])):
    dataFile=physionetFile+f"{recording_num}_{timespan}secs_ecg.csv"
    data=torch.tensor(np.array(pd.read_csv(dataFile)),dtype=torch.float32)

    #print(data.type())

    predictions=torch.argmax(model.forward(data)[:-2],dim=-1) #Get the class positions

    #Get the prediction counts for each type:
    counts=np.zeros(4)
    for i in range(4):
        counts[i]=int(torch.sum(predictions==i))

    overall_prediction=np.argmax(counts)

    #Now we compare the overall prediction to the one that is in the annotations

    true_annotation=np.array(annotations["Their_label"])[anno_index]

    true_annotation=np.arange(3)[true_annotation==posLabels]

    #Add 1 to the confusion matrix in the right box
    ConfMat[true_annotation,overall_prediction]=ConfMat[true_annotation,overall_prediction]+1 #The row is the true value, the column is the predicted

print(f"Confusion Matrix: {ConfMat}")
