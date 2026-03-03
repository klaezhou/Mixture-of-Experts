import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
import logging
import torch.nn.init as init


class MLP(nn.Module):
    """
    Expert network class. Using Tanh as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - hidden_size (int): The size of the hidden layer.
    """
    def __init__(self,args):
        super(MLP, self).__init__()
        self.depth=getattr(args, "mlp_depth", 2)
        input_size=getattr(args, "input_size", 2)
        hidden_size=getattr(args, "mlp_hidden_size", 50)
        output_size=getattr(args, "output_size", 1)
        activation=nn.Tanh()
        self.activation=activation
        layer_list = []
        layer_list.append(nn.Linear(input_size,hidden_size))  #[I,H]
        for i in range(self.depth-1):
            layer_list.append(nn.Linear(hidden_size,hidden_size))  #[H,H]
        layer_list.append(nn.Linear(hidden_size,output_size))  #[H,I] 
        self.net = nn.ModuleList(layer_list)
        self._init_weights()
        self._report_trainable()
        
    def _report_trainable(self):
            total = 0
            print("=== MLP Trainable parameters ===")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    n = p.numel()
                    total += n
            print(f"MLP trainable params: {total}")
            
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)
        
    @torch.no_grad()
    def forward_penultimate(self, x):
        """
        Return the activation after the last hidden layer (before final linear).
        Shape: [N, hidden_size]
        """
        y = x
        for layer in self.net[:-1]:
            y = self.activation(layer(y))
        return y
    def forward(self, y):
        for i, layer in enumerate(self.net[:-1]):
            y = layer(y)
            y = self.activation(y)
        y = self.net[-1](y)
        return y
    
# feature getter
@torch.no_grad()
def mlp_penultimate_getter(model, x):
    """
    Get the penultimate layer activations from an MLP model.

    Parameters:
    - model (MLP): The MLP model instance.
    - x (torch.Tensor): Input tensor of shape [N, input_size].

    Returns:
    - torch.Tensor: Activations from the penultimate layer of shape [N, hidden_size].
    """
    return model.forward_penultimate(x)