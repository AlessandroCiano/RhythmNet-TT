import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuralMemory(nn.Module):
    def __init__(self, input_shape, hidden_units, alpha=0.1, nu = 0.9, theta = 0.3):
        super(NeuralMemory,self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.accumulated_surprise = None
        self.alpha = alpha # past memory to forget
        self.nu = nu # surprise decay (how quickly past “surprise” fades)
        self.theta = theta # momentary surprise scaling
        self.BuildModule()

    def BuildModule(self):
        # If input_shape is an int, assume it represents the feature dimension and add a batch dimension of 1.
        if isinstance(self.input_shape, int):
            x = torch.zeros(1, self.input_shape)
        else:
            x = torch.zeros(self.input_shape)
        print("Building module with input shape ", x.shape)

        self.layer_dict=nn.ModuleDict()
        self.layer_dict["Input_Layer"] = nn.Linear(x.shape[-1], self.hidden_units)
        x = self.layer_dict["Input_Layer"].forward(x)
        print("shape after 1st layer: ", x.shape)
        self.layer_dict["bn0"] = nn.BatchNorm1d(x.shape[1])
        x = self.layer_dict["bn0"].forward(x)

        out_features = self.input_shape if isinstance(self.input_shape, int) else self.input_shape[-1]
        self.layer_dict["Output_Layer"] = nn.Linear(x.shape[-1], out_features)
        x = self.layer_dict["Output_Layer"](x)
        print("shape after final layer: ", x.shape)
        return x
    
    def forward(self, x):
        out = self.layer_dict["Input_Layer"].forward(x)
        out = F.relu(self.layer_dict["bn0"].forward(out))
        out = self.layer_dict["Output_Layer"].forward(out)
        return out
    
    def get_params(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def surprise_score(self, k, v):
        prediction = self.forward(k)
        loss = F.mse_loss(prediction, v)

        params = self.get_params()
        surprise_s = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        return surprise_s

    def update_memory(self, surprise):
        """
        Update memory parameters: w -> (1-alpha)w + alpha*surprise
        This is a differentiable version that allows gradients to flow through
        """
        updated_params = {}
        for (name, param), grad in zip(self.named_parameters(), surprise):
            updated_params[name] = (1 - self.alpha) * param + self.alpha * grad
        return updated_params

    def forward_inference(self, k, q, v):
        '''
        Used to train the model with post attention sequence
        k: keys (input)
        q: query
        v: values (target)
        '''
        momentary_surprise = self.surprise_score(k, v)

        # Initialize accumulated_surprise if it's the first call
        if self.accumulated_surprise is None:
            self.accumulated_surprise = [torch.zeros_like(grad) for grad in momentary_surprise]
        
        # Update accumulated_surprise element-wise
        new_accumulated = []
        for i, (acc_surp, mom_surp) in enumerate(zip(self.accumulated_surprise, momentary_surprise)):
            new_acc = self.nu * acc_surp - self.theta * mom_surp
            new_accumulated.append(new_acc)
        self.accumulated_surprise = new_accumulated

        updated_params = self.update_memory(self.accumulated_surprise)

        # differentiable forward pass with updated parameters (Non-persistent updated parameters)
        output = torch.func.functional_call(self, updated_params, q)

        '''
        # Persistent updated parameters
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.data.copy_(updated_params[name].data)  # Non-differentiable update (not affecting backpropagation)
        '''
        self.updated_params_cache = {name: param.clone().detach() for name, param in updated_params.items()}
        
        return output
    
    def apply_cached_updates(self):
        """
        Call this method after backward() has been called on the loss
        to apply the parameter updates without breaking the graph
        """
        if hasattr(self, 'updated_params_cache'):
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if name in self.updated_params_cache:
                        param.data.copy_(self.updated_params_cache[name].data)
            # Clear the cache
            del self.updated_params_cache

    def reset_computational_history(self):
        # Break the computational graph history
        if self.accumulated_surprise is not None:
            self.accumulated_surprise = [
                s.clone().detach() for s in self.accumulated_surprise
            ]

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class TryBackprop(nn.Module):
    def __init__(self, input_shape, hidden_units):
        super(TryBackprop,self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.BuildModule()
    
    def BuildModule(self):
        # two linear layers with NeuralMemory in the middle, quries, keys and values
        x = torch.zeros(self.input_shape)
        self.layer_dict=nn.ModuleDict()
        self.layer_dict["Queries"] = nn.Linear(x.shape[-1], self.hidden_units)
        q = self.layer_dict["Queries"].forward(x)
        self.layer_dict["Keys"] = nn.Linear(x.shape[-1], self.hidden_units)
        k = self.layer_dict["Keys"].forward(x)
        self.layer_dict["Values"] = nn.Linear(x.shape[-1], self.hidden_units)
        v = self.layer_dict["Values"].forward(x)

        self.layer_dict["Neural_Memory"] = NeuralMemory(input_shape=q.shape, hidden_units=self.hidden_units)
        out = self.layer_dict["Neural_Memory"].forward_inference(k, q, v)

        self.layer_dict["Output_Layer"] = nn.Linear(out.shape[-1], self.input_shape[-1])
        out = self.layer_dict["Output_Layer"].forward(out)
        return out
    
    def forward(self, x):
        q = self.layer_dict["Queries"].forward(x)
        k = self.layer_dict["Keys"].forward(x)
        v = self.layer_dict["Values"].forward(x)
        out = self.layer_dict["Neural_Memory"].forward_inference(k, q, v)
        out = self.layer_dict["Output_Layer"].forward(out)
        return out