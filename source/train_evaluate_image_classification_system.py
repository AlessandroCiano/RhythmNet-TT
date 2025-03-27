import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from model_architectures import FullyConnectedNetwork, Quite_Big_Model, Quite_Big_Titan_Model
from arg_extractor import get_args
import numpy as np
import torch
from experiment_builder import ExperimentBuilder

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed
ecg_data = pd.read_csv(args.dataset_path)

is_titan=args.is_titan
is_pretrain=args.is_pretrain

labelcol=int(ecg_data.select_dtypes(include=["object"]).columns[0])+1 #First column with a label in

X=ecg_data.iloc[:,1:labelcol].values #All columns up to the labels

Yvals=np.array(ecg_data.iloc[:,labelcol:].values) #All the labels

#Depending on pre-training or not, we need a different one-hot encoding
if not args.is_pretrain: #If we are not pretraining, then we do the 5 class one-hot
    def One_Hot(Input):
        """Function to one-hot encode a 1xn numpy array, returns an nx4 encoded variable"""
        np.putmask(Input,(Input=="L")+(Input=="R")+(Input=="j")+(Input=="e"),"N")#Replace the multiple labels that mean `N` with N
        np.putmask(Input,(Input=="A")+(Input=="a")+(Input=="S")+(Input=="J"),"S")#Replace the multiple labels that mean `S` with S
        np.putmask(Input,(Input=="!")+(Input=="V")+(Input=="E")+(Input=="[")+(Input=="]"),"V")#Replace the multiple labels that mean `V` with V
        np.putmask(Input,(Input!="N")*(Input!="S")*(Input!="V")*(Input!="F"),"NA")#Replace the multiple labels that should be unclassified with 0
        Classes=np.array(["N","S","V","F","NA"])[:,np.newaxis] #N,S,V,F and empty classes
        return Classes.T==Input[:,np.newaxis]

    def ConvertClassandPos(Yinput):
        """
        Converts our classification data into something the model can predict

        Inputs
        ------
        Yinput: An array including two columns - one of the string classification, another with the position (size: [n,2])

        Returns
        -------
        An array of size [n,1,6] - one hot encoded with position at end (the 6)
        """
        Onehotted=One_Hot(Yinput[:,0])
        Yret=np.append(Onehotted,Yinput[:,1][:,np.newaxis]/labelcol,axis=1) #divide by labelcol, to make the positions range from 0 to 1
        return Yret[:,np.newaxis,:]
    num_classes=5
else:
    def One_Hot(Input):
        """Function to one-hot encode a 1xn numpy array, returns an nx3 encoded variable"""
        np.putmask(Input, Input == "0", "NA")  # Replace '0' with 'NA'
        np.putmask(Input, (Input != "N") & (Input != "AFIB") & (Input != "NA"), "NA")  # Ensure only valid labels
        Classes = np.array(["N", "AFIB", "NA"])[:, np.newaxis]  # Defined categories
        return Classes.T == Input[:, np.newaxis]

    def ConvertClassandPos(Yinput):
        """
        Converts our classification data into something the model can predict

        Inputs
        ------
        Yinput: An array including two columns - one of the string classification, another with the position (size: [n,2])
        Returns
        -------
        An array of size [n,1,4] - one hot encoded with position at end (the 4)
        """
        Onehotted = One_Hot(Yinput[:, 0])
        Yret = np.append(Onehotted, Yinput[:, 1][:, np.newaxis], axis=1)  # Keep position unchanged
        return Yret[:, np.newaxis, :]
    num_classes=3




Y=ConvertClassandPos(Yvals[:,:2]) #Convert classification

for i in range(2,int(Yvals.shape[1]),2):
    #Loop through all possible classifications, and convert them
    Y=np.append(Y,ConvertClassandPos(Yvals[:,(i):(i+2)]),axis=1)
    
"""# Separate features and target
X = ecg_data.iloc[:, 1:-2].values  # All columns except the last one
# y = ecg_data.iloc[:, -1].str.strip().replace({'N': 0, 'A': 1}).values  # Last column, remove spaces and replace labels

y = (ecg_data.iloc[:, -2].astype(str).str.strip() != 'N').astype(int).values
y2 = ecg_data.iloc[:, -1]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)"""

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.3, random_state=args.seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.seed)


# Ensure that the numpy arrays are of type float32 for X and float32 for y
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

#Normalise Data
def NormData(train,val,test):
    mu=np.mean(train)
    sig=np.std(train)
    train=(train-mu)/sig
    val=(val-mu)/sig
    test=(test-mu)/sig
    return train,val,test

X_train,X_val,X_test=NormData(X_train,X_val,X_test)


# Create DataLoaders
train_data = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=args.batch_size, shuffle=True)
val_data = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=args.batch_size, shuffle=False)
test_data = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=args.batch_size, shuffle=False)
# Define the model
# model = FullyConnectedNetwork(input_features=args.num_features, hidden_units=64, output_classes=args.num_classes)
if is_titan:#Check if the model is the titan model
    model=Quite_Big_Titan_Model(input_shape=[args.batch_size,1,labelcol-1],
                                d_model=6,
                                transformer_heads=args.transformer_heads,
                                hidden_units=args.hidden_units,
                                num_classes=num_classes,
                                N_q=Yvals.shape[1]//2,
                                phi=args.phi,
                                nm_hu=args.nm_hu,
                                nm_kqv_size=args.nm_kqv_size,
                                persistent_dim=args.pers_dim,
                                alpha=args.alpha,
                                nu=args.nu,
                                theta=args.theta)
else:
    model=Quite_Big_Model(input_shape=[args.batch_size,1,labelcol-1],
                        d_model=6,
                        transformer_heads=args.transformer_heads,
                        hidden_units=args.hidden_units,
                        num_classes=num_classes,
                        N_q=Yvals.shape[1]//2
                        )

# model=Quite_Big_Model(input_shape=[args.batch_size,1,args.num_features],
#                       d_model=6,
#                       transformer_heads=args.transformer_heads,
#                       hidden_units=args.hidden_units,
#                       num_classes=5,
#                       )

# model = Quite_Big_Titan_Model(input_shape=(args.batch_size, 1, args.num_features), hidden_units=64, output_classes=3)

# Build and run the experiment
experiment = ExperimentBuilder(network_model=model,
                                experiment_name=args.experiment_name,
                                num_epochs=args.num_epochs,
                                weight_decay_coefficient=args.weight_decay_coefficient,
                                learning_rate=args.learning_rate,
                                use_gpu=args.use_gpu,
                                continue_from_epoch=args.continue_from_epoch,
                                train_data=train_data,
                                val_data=val_data,
                                test_data=test_data,
                                is_titan=is_titan,
                                is_pretrain=is_pretrain,
                                actually_continue=args.actually_continue)
experiment_metrics, test_metrics = experiment.run_experiment()
