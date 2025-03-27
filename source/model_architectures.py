import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation
import math
from titan import NeuralMemory


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_features, hidden_units, output_classes):
        """
        Initializes a fully connected network for tabular data.
        :param input_features: Number of input features (columns in the dataset).
        :param hidden_units: Number of units in the hidden layer.
        :param output_classes: Number of output classes (e.g., N and A).
        """
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_units)  # First layer
        self.fc2 = nn.Linear(hidden_units, output_classes)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        self.softmax = nn.Softmax(dim=1)  # For multi-class classification

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Predicted logits.
        """
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

    def reset_parameters(self):
        """
        Resets the network parameters for reinitialization.
        """
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

class FeedForwardModule(nn.Module):
    def __init__(self, input_shape,hidden_units,num_classes,finallayer="softmax"):
        """
        Initializes a feed-forward network that can be either regression or classification
        
        Parameters
        ----------
        input_shape
            Shape of the input
        hidden_units
            Number of hidden units in the hidden layer
        num_classes
            Number of outputs
        finallayer
            Determines the type of final layer in the model - softmax is classification, if changed just produces regression
        """
        super(FeedForwardModule,self).__init__()
        self.input_shape=input_shape
        self.hidden_units=hidden_units
        self.num_classes=num_classes
        self.finallayer=finallayer
        self.BuildModule()
    def BuildModule(self):
        x=torch.zeros(self.input_shape)
        print("Building module with input shape ", x.shape)
        self.layer_dict=nn.ModuleDict()
        self.layer_dict["FF0"]=nn.Linear(x.shape[-1],self.hidden_units)
        x=self.layer_dict["FF0"].forward(x)
        print("shape after 1st layer: ",x.shape)
        self.layer_dict["bn0"]=nn.BatchNorm1d(x.shape[1])
        x=self.layer_dict["bn0"].forward(x)
        print("shape after batch normalisation layer: ", x.shape)
        output_num=2 #2 if we are classifying positions
        if self.finallayer=="softmax":
            #Change final layer size to number of classes, if this is a classification task
            output_num=self.num_classes
        self.layer_dict["FF1"]=nn.Linear(x.shape[-1],output_num)
        x=self.layer_dict["FF1"].forward(x)
        print("shape after final layer: ",x.shape)
        print("final x shape: ",x.shape)
        return x
    def forward(self,input):
        x=input
        x=self.layer_dict["FF0"].forward(x)
        x=F.relu(self.layer_dict["bn0"].forward(x))
        x=self.layer_dict["FF1"].forward(x)

        if self.finallayer=="softmax":
            x=F.softmax(x,dim=2)
        return x
    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class MultipleOutputFFN(nn.Module):
    def __init__(self, input_dim,hidden_units,num_classes):
        """
        Initializes a cnn backbone for the feature extraction for the transformer
        
        Parameters
        ----------
        input_shape
            Shape of the input
        hidden_units
            Number of hidden units used 
        num_classes
            Number of classes model has to predict (N,A etc.)
        """
        super(MultipleOutputFFN,self).__init__()
        self.input_dim=input_dim
        self.hidden_units=hidden_units
        self.output_dim=num_classes
        self.BuildModule()
    def BuildModule(self):
        x=torch.zeros(self.input_dim)
        self.layer_dict=nn.ModuleDict()
        self.layer_dict["Locations"]=FeedForwardModule(self.input_dim,self.hidden_units,self.output_dim,finallayer="not")
        out1=self.layer_dict["Locations"].forward(x)
        print("output1 shape: ",out1.shape)
        self.layer_dict["Classification"]=FeedForwardModule(self.input_dim,self.hidden_units,self.output_dim)
        out2=self.layer_dict["Classification"].forward(x)
        print("output2 shape: ",out2.shape)
        outfinal=torch.cat([out2,out1],dim=2)
        print("output final shape: ",outfinal.shape)
        return outfinal
    def forward(self,input):
        x=input
        out1=self.layer_dict["Locations"].forward(x)
        out2=self.layer_dict["Classification"].forward(x)
        outfinal=torch.cat([out2,out1],dim=2)
        return outfinal
    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class SqueezeExcite(nn.Module):
    def __init__(self, input_shape,reduction=16):
        """
        Initializes a squeeze and excitement module for the bottleneck 
        
        Parameters
        ----------
        input_shape
            Shape of the input
        reduction
            Size of reduction within the squeeze part
        """
        super(SqueezeExcite,self).__init__()
        self.input_shape=input_shape
        self.red=reduction
        self.BuildModule()
    def BuildModule(self):
        out=torch.zeros(self.input_shape)
        self.layer_dict=nn.ModuleDict()
        bottleneck=max(out.shape[1]//self.red,1)
        self.layer_dict["Avg_Pooling"]=nn.AvgPool1d(1)
        self.layer_dict["Conv1D"]=nn.Conv1d(out.shape[1],bottleneck,kernel_size=1,padding=0)
        self.layer_dict["Relu"]=nn.ReLU(True)
        self.layer_dict["Conv1d2"]=nn.Conv1d(bottleneck,out.shape[1],kernel_size=1,padding=0)
        self.layer_dict["Sigmoid"]=nn.Sigmoid()
    def forward(self,input):
        out=input
        out=self.layer_dict["Avg_Pooling"].forward(out)
        out=self.layer_dict["Conv1D"].forward(out)
        out=self.layer_dict["Relu"].forward(out)
        out=self.layer_dict["Conv1d2"].forward(out)
        out=self.layer_dict["Sigmoid"].forward(out)
        return out*input
    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass




class SEModule(nn.Module):
    def __init__(self, channels, divide=4):
        super(SEModule, self).__init__()
        bottleneck = channels //  divide
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x
    def reset_parameters(self):
        self.se.reset_parameters()

class Bottleneck(nn.Module):
    def __init__(self, input_shape, k, c, n, s):
        """
        Initializes a bottleneck module
        
        Parameters
        ----------
        input_shape
            Shape of the input
        k
            Size of final channel output (hyper-parameter)
        c
            Output channels of bottleneck layer
        n
            N
        s
            Stride length used in first layer
        """
        super(Bottleneck,self).__init__()
        self.input_shape = input_shape
        self.k = k
        self.c = c
        self.n = n
        self.s = s
        self.build_module()

    def build_module(self):

        print("------------------------------------------")

        self.layer_dict = nn.ModuleDict()

        x = torch.zeros((self.input_shape))
        out = x
        print("\n bottleneck 1 input shape: ",out.shape)
        self.layer_dict['fist_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.c, kernel_size = 1, stride = self.s, padding = 0)
        self.layer_dict['bn0'] = nn.BatchNorm1d(num_features= self.c)
        
        out = self.layer_dict['fist_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn0'].forward(out)
        out = F.relu(out)
        print("\n bottleneck 1st layer ouput shape: ",out.shape)

        self.layer_dict['depthwise_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.k * out.shape[1], groups = out.shape[1], kernel_size = 3, stride = 1, padding = 0)
        


        out = self.layer_dict['depthwise_bottleneck_conv1d'].forward(out)
        print("\n bottleneck depthwise 2nd layer ouput shape: ",out.shape)
        self.layer_dict['bn1'] = nn.BatchNorm1d(num_features= out.shape[1])

        out = self.layer_dict['bn1'].forward(out)
        out = F.relu(out)
        
        self.layer_dict['last_bottleneck_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.c, kernel_size = 1, stride = 1, padding = 1)
        self.layer_dict['bn2'] = nn.BatchNorm1d(num_features= self.c)
        
        out = self.layer_dict['last_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn2'].forward(out)


        print("Output before SE",out.shape)

        
        self.layer_dict['SE'] = SqueezeExcite(input_shape=out.shape,reduction=4) #Homemade squeeze excite function
        out = self.layer_dict['SE'](out)

        
        
        print("\n bottleneck ouput shape: ",out.shape)

        if(out.shape == x.shape):
            out = out + x

        return out
    
    def forward(self, x):
        out = x

        out = self.layer_dict['fist_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn0'].forward(out)
        out = F.relu(out)

        out = self.layer_dict['depthwise_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn1'].forward(out)
        out = F.relu(out)

        out = self.layer_dict['last_bottleneck_conv1d'].forward(out)
        out = self.layer_dict['bn2'].forward(out)
        
        
        out = self.layer_dict['SE'](out)
        

        if(out.shape == x.shape):
            out = out + x


        return out
    
    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass





class cnnBackbone(nn.Module):
    def __init__(self, input_shape, d_model,phi=1):
        """
        Initializes a cnn backbone for the feature extraction for the transformer
        
        Parameters
        ----------
        input_shape
            Shape of the input
        d_model
            Size of final channel output (hyper-parameter)
        phi
            The phi parameter mentioned in the original paper - to shrink the model size
        """
        super(cnnBackbone, self).__init__()
        self.input_shape = input_shape
        #self.hidden_units = hidden_units
        #self.output_classes = output_classes
        self.d_model = d_model
        self.phi=phi
        self.build_module()

    def build_module(self):
        self.layer_dict = nn.ModuleDict()
        x = torch.zeros(self.input_shape)
        out = x

        # first conv
        self.layer_dict['fist_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = 1, kernel_size = 1, stride = 2, padding = 0)
        
        out = self.layer_dict['fist_conv1d'].forward(out)

        self.layer_dict['bn0_outside'] = nn.BatchNorm1d(num_features= out.shape[1])
        out = self.layer_dict['bn0_outside'].forward(out)  #ERRROOOORRRRR
        out = F.relu(out)

        # bottlenecks
        bottleneckRepeats = [1, 2, 2, 2, 2, 1]
        strideList=[2, 2, 2, 2, 2, 1]
        cList = [8, 16, 32, 64, 128, 256]
        for i in range(0, 6):
            ctr = strideList[i]
            while bottleneckRepeats[i] > 0:
                print(f'Bottleneck_{i}_{bottleneckRepeats[i]}')
                self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats[i]}'] = Bottleneck(input_shape=out.shape, k = 6, c = int(self.phi*cList[i]), n = bottleneckRepeats[i], s = ctr )
                ctr = ctr - 1 # creates the bottleneck
                #runs the values through the created bottleneck as it was created
                out = self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats[i]}'].forward(out)
                bottleneckRepeats[i]-=1

        # last conv
        self.layer_dict['last_conv1d'] = nn.Conv1d(in_channels = out.shape[1], out_channels = self.d_model, kernel_size = 1, stride = 1, padding = 0)
        
        out = self.layer_dict['last_conv1d'].forward(out)

        self.layer_dict['bn1_outside'] = nn.BatchNorm1d(num_features= out.shape[1])
        out = self.layer_dict['bn1_outside'].forward(out)
        out = F.relu(out)

        print("Output shape: ",out.shape)

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Predicted logits.
        """
        out = self.layer_dict['fist_conv1d'].forward(x)
        out = self.layer_dict['bn0_outside'].forward(out)
        out = F.relu(out)

        bottleneckRepeats = [1, 2, 2, 2, 2, 1]
        # cList = [8, 16, 32, 64, 128, 256]

        for i in range(0, 6):
            while bottleneckRepeats[i] > 0:
                out = self.layer_dict[f'Bottleneck_{i}_{bottleneckRepeats[i]}'].forward(out)
                bottleneckRepeats[i]-=1


        # last conv
        out = self.layer_dict['last_conv1d'].forward(out)
        out = self.layer_dict['bn1_outside'].forward(out)
        out = F.relu(out)

        return out

    def reset_parameters(self):
        """
        Resets the network parameters for reinitialization.
        """
        for layerkey in self.layer_dict.keys():
            self.layer_dict[layerkey].reset_parameters()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initialize the PositionalEncoding module.

        Args:
            d_model: The number of expected features in the input.
            max_len: The maximum length of the input sequence.
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # Create a matrix to hold the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension for easier integration
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input.
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model].
        Returns:
            Tensor with positional encodings added to the input.
        """
        x = x + self.pe[:x.size(0), :]
        return x

    def reset_parameters(self):
        """
        Resets the parameters (if any learnable parameters are added in future versions).
        """
        self.pe.zero_()


# Transformer adapted for our task
class ECGTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_queries, num_encoder_layers, num_decoder_layers, max_seq_len=5000):
        super(ECGTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_seq_len = max_seq_len
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_seq_len)

        # Transformer components
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        
        # Learnable object queries
        self.object_queries = nn.Parameter(torch.rand(num_queries, d_model))
        
    def forward(self, features):
        """
        :param features: Extracted features from ECG signals (batch_size, seq_len, d_model)
        :return: Predicted classes and positions of heartbeats
        """
        # Add positional encoding
        features = self.positional_encoding(features)  # Shape: (batch_size, seq_len, d_model)


        # Encoder
        encoded_features = self.encoder(features)  # Shape: (batch_size, seq_len, d_model)
        
        # Prepare object queries for each sample in the batch
        batch_size = features.size(0)
        queries = self.object_queries.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: (batch_size, num_queries, d_model)
        
        # Decoder
        decoded_features = self.decoder(queries, encoded_features)  # Cross-attention with encoder output
        
        return decoded_features


class Quite_Big_Model(nn.Module):
    def __init__(self, input_shape ,d_model,transformer_heads,hidden_units,num_classes,N_q=6,phi=1):
        """
        Initializes a quite big model - cnn backbone into transformer into fully connected networks
        
        Parameters
        ----------
        input_shape
            Shape of the input
        d_model
            Dimension of model hyper-parameter
        transformer_heads
            Number of heads in transformer layer - needs to be a factor of d_model
        hidden_units
            Number of hidden units in fully connected networks
        num_classes
            Number of classes to predict
        phi
            Hyperparameter indicating the scale of the CNN backbone
        """
        super(Quite_Big_Model,self).__init__()
        self.input_shape=input_shape
        self.d_model=d_model
        if d_model/transformer_heads!=d_model//transformer_heads:
            #If the d_model is not a multiple of transformer heads, return an error and exit
            raise ValueError("Error: the d_model is not a multiple of the number of transformer heads")
        self.num_heads=transformer_heads
        self.hidden_units=hidden_units
        self.num_classes=num_classes
        self.N_q=N_q
        self.phi=phi

        self.build_module()
        
    def build_module(self):
        out=torch.zeros(self.input_shape)
        self.layer_dict=nn.ModuleDict()
        print(f"Initialising Quite A Big Model with input shape {self.input_shape}")

        #CNN Backbone
        self.layer_dict["Cnn_Backbone"]=cnnBackbone(out.shape,self.d_model,phi=self.phi)

        out=self.layer_dict["Cnn_Backbone"].forward(out)

        print("Backbone Output Shape: ",out.shape)

        #Transformer Layer - I'm not sure if the d_model is meant to be the same as our d_model, but it doesn't work if I try that
        ##self.layer_dict["Transformer"]=nn.Transformer(d_model=self.d_model,nhead=self.num_heads,batch_first=True)
        '''
        self.layer_dict["Transformer"]=nn.Transformer(d_model=out.shape[-1],nhead=self.num_heads,batch_first=True)

        out=self.layer_dict["Transformer"].forward(out,out)
        '''
        out = out.permute(0, 2, 1)  # (batch_size, 17, d_model)

        self.layer_dict["Transformer"]=ECGTransformer(d_model=out.shape[-1],num_heads=self.num_heads,num_queries=out.shape[1],num_encoder_layers=4,num_decoder_layers=4)

        out=self.layer_dict["Transformer"].forward(out)

        print("Transformer Output Shape: ",out.shape)

        #Conv for N_q shape
        self.layer_dict["Conv1d_N_q"]=nn.Conv1d(out.shape[1], self.N_q, kernel_size=1, padding=0)

        out=self.layer_dict["Conv1d_N_q"].forward(out)

        #Final module
        self.layer_dict["FullyConnected"]=MultipleOutputFFN(out.shape,self.hidden_units,self.num_classes)

        out=self.layer_dict["FullyConnected"].forward(out)

        print("Final Ouput Shape: ",out.shape)
        
    def forward(self,Input):
        Input=Input.unsqueeze(1) #Need to add this as the data loader does not add the channel term
        #if Input.shape!=torch.zeros(self.input_shape).shape:
        #    #Just check we are giving the right input into the model
        #    raise ValueError(f"Error: Input supplied ({Input.shape}) is not the same size as intialised ({self.input_shape})")
        out=self.layer_dict["Cnn_Backbone"].forward(Input)
        out=out.permute(0,2,1)
        out=self.layer_dict["Transformer"].forward(out)#Not sure if this is the right way to forward pass the transformer
        out=self.layer_dict["Conv1d_N_q"].forward(out)
        out=self.layer_dict["FullyConnected"].forward(out)

        return out
    def reset_parameters(self):
        self.layer_dict["Cnn_Backbone"].reset_parameters()
        #self.layer_dict["Transformer"].reset_parameters()
        self.layer_dict["Conv1d_N_q"].reset_parameters()
        self.layer_dict["FullyConnected"].reset_parameters()


class Quite_Big_Titan_Model(nn.Module):
    def __init__(self, input_shape ,d_model,transformer_heads,hidden_units,num_classes,N_q=6,phi=1,
                 nm_hu = 128,nm_kqv_size = 64,persistent_dim = 32,alpha = 0.1,nu = 0.9,theta = 0.3):
        """
        Initializes a quite big model - cnn backbone into transformer into fully connected networks
        
        Parameters
        ----------
        input_shape
            Shape of the input
        d_model
            Dimension of model hyper-parameter
        transformer_heads
            Number of heads in transformer layer - needs to be a factor of d_model
        hidden_units
            Number of hidden units in fully connected networks
        num_classes
            Number of classes to predict
        """
        super(Quite_Big_Titan_Model,self).__init__()
        self.input_shape=input_shape
        self.d_model=d_model
        if d_model/transformer_heads!=d_model//transformer_heads:
            #If the d_model is not a multiple of transformer heads, return an error and exit
            raise ValueError("Error: the d_model is not a multiple of the number of transformer heads")
        self.num_heads=transformer_heads
        self.hidden_units=hidden_units
        self.num_classes=num_classes
        self.N_q=N_q
        self.phi=phi

        #to be made into arg_extractor later!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.nm_hu = nm_hu
        self.nm_kqv_size = nm_kqv_size
        self.persistent_dim = persistent_dim
        self.alpha = alpha # past memory to forget
        self.nu = nu # surprise decay (how quickly past “surprise” fades)
        self.theta = theta # momentary surprise scaling

        self.build_module()
        
    def build_module(self):
        out=torch.zeros(self.input_shape)
        self.layer_dict=nn.ModuleDict()
        print(f"Initialising Quite A Big Model with input shape {self.input_shape}")

        #CNN Backbone
        self.layer_dict["Cnn_Backbone"]=cnnBackbone(out.shape,self.d_model,phi=self.phi)

        out=self.layer_dict["Cnn_Backbone"].forward(out)

        print("Backbone Output Shape: ",out.shape)

        '''
        self.layer_dict["Transformer"]=nn.Transformer(d_model=out.shape[-1],nhead=self.num_heads,batch_first=True)

        out=self.layer_dict["Transformer"].forward(out,out)
        '''
        out = out.permute(0, 2, 1)  # (batch_size, 17, d_model)
        lin_shape = out.shape[-1]

        #first pass neural memory
        self.layer_dict["Queries"] = nn.Linear(out.shape[-1], self.nm_kqv_size) # query layer initialized
        q_first = self.layer_dict["Queries"].forward(out)
        self.layer_dict["Neural_Memory"] = NeuralMemory(input_shape=q_first.shape, hidden_units=self.nm_hu, alpha=self.alpha, nu=self.nu, theta=self.theta)
        out_first_nm = self.layer_dict["Neural_Memory"].forward(q_first)
        
        #persistent memory
        self.persistent_memory = nn.Parameter(torch.zeros(1, out.shape[1], self.persistent_dim))
        persistent_memory_expanded = self.persistent_memory.expand(out.shape[0], -1, -1) # expand to batch size
        out = torch.cat((out, out_first_nm, persistent_memory_expanded), dim=2)
        
        #transformer
        print("Input shape of Transformer: ", out.shape)
        self.layer_dict["Transformer"]=ECGTransformer(d_model=out.shape[-1],num_heads=self.num_heads,num_queries=out.shape[1],num_encoder_layers=4,num_decoder_layers=4)
        out=self.layer_dict["Transformer"].forward(out)
        print("Transformer Output Shape: ",out.shape)
        self.layer_dict["Linear_reducer"] = nn.Linear(out.shape[-1], lin_shape) # reduction
        out = self.layer_dict["Linear_reducer"].forward(out)

        #second pass neural memory
        self.layer_dict["Keys"] = nn.Linear(out.shape[-1], self.nm_kqv_size) # key layer initialized
        k_second = self.layer_dict["Keys"].forward(out)
        self.layer_dict["Values"] = nn.Linear(out.shape[-1], self.nm_kqv_size) # value layer initialized
        q_second = self.layer_dict["Queries"].forward(out)
        v_second = self.layer_dict["Values"].forward(out)
        out_second_nm = self.layer_dict["Neural_Memory"].forward_inference(k_second, q_second, v_second)

        # dot product out with out_second_nm
        self.layer_dict['Linear_trans_shape'] = nn.Linear(self.nm_kqv_size, lin_shape) # back to original shape
        out_second_nm = self.layer_dict['Linear_trans_shape'].forward(out_second_nm)
        out = torch.bmm(out, out_second_nm.transpose(2, 1))
        print("Output shape after dot product: ", out.shape)

        #Conv for N_q shape
        self.layer_dict["Conv1d_N_q"]=nn.Conv1d(out.shape[1], self.N_q, kernel_size=1, padding=0)
        out=self.layer_dict["Conv1d_N_q"].forward(out)

        #Final module
        self.layer_dict["FullyConnected"]=MultipleOutputFFN(out.shape,self.hidden_units,self.num_classes)

        out=self.layer_dict["FullyConnected"].forward(out)

        print("Final Ouput Shape: ",out.shape)

        #Set the accumulated surprise back to None to allow it to initialise again
        self.layer_dict["Neural_Memory"].accumulated_surprise =None #This is needed so the accumulated surprise is created on the correct device
        
    def forward(self,Input):
        Input=Input.unsqueeze(1) #Need to add this as the data loader does not add the channel term
        #if Input.shape!=torch.zeros(self.input_shape).shape:
        #    #Just check we are giving the right input into the model
        #    raise ValueError(f"Error: Input supplied ({Input.shape}) is not the same size as intialised ({self.input_shape})")
        out=self.layer_dict["Cnn_Backbone"].forward(Input)
        out=out.permute(0,2,1)

        q_first = self.layer_dict["Queries"].forward(out)
        out_first_nm = self.layer_dict["Neural_Memory"].forward(q_first)

        persistent_memory_expanded = self.persistent_memory.expand(out.shape[0], -1, -1) # expand to batch size
        out = torch.cat((out, out_first_nm, persistent_memory_expanded), dim=2)

        out=self.layer_dict["Transformer"].forward(out) #Not sure if this is the right way to forward pass the transformer
        out = self.layer_dict["Linear_reducer"].forward(out) # reduction
        
        k_second = self.layer_dict["Keys"].forward(out)
        q_second = self.layer_dict["Queries"].forward(out)
        v_second = self.layer_dict["Values"].forward(out)

        out_second_nm = self.layer_dict["Neural_Memory"].forward_inference(k_second, q_second, v_second)
        out_second_nm = self.layer_dict['Linear_trans_shape'].forward(out_second_nm) # back to original shape

        out = torch.bmm(out, out_second_nm.transpose(2, 1))

        out=self.layer_dict["Conv1d_N_q"].forward(out)
        out=self.layer_dict["FullyConnected"].forward(out)

        return out
    def reset_parameters(self):
        self.layer_dict["Cnn_Backbone"].reset_parameters()
        #self.layer_dict["Transformer"].reset_parameters()
        self.layer_dict["Conv1d_N_q"].reset_parameters()
        self.layer_dict["FullyConnected"].reset_parameters()