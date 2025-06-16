# =============================================================================
# Mixture of Experts Module for Functional Approximation
# -----------------------------------------------------------------------------
# Author : klae-zhou
# Date   : 2025-06-05
# -----------------------------------------------------------------------------
# Description:
# This module implements a sparsely-gated Mixture of Experts (MoE) layer
# designed for function approximation tasks. The MoE architecture consists of:
#   - A gating network that selects a sparse subset of experts per input.
#   - Multiple expert networks (typically MLPs) that process the input.
#   - Aggregation of expert outputs based on gating decisions.
#
# Key Features:
#   - Top-k gating (default:top-2)
#   - Expert selection per token/sample
#   - Modular design suitable for PyTorch integration
# Architechture:
#   Input->Gating  Network*Experts->FCNNs->Output
# Usage:
#   - Plug into a larger neural network as a learnable, conditional MLP layer
#   - Designed for regression, modeling nonlinear functions, or approximating
#     high-dimensional mappings with sparse expert activation
#
# Dependencies:
#   - torch
#   - torch.nn
#   - torch.nn.functional
#
# Notes:
#   - Expert load balancing and capacity limiting are optional features.
#   - This implementation assumes single-device execution. For distributed
#     training, refer to DeepSpeed-MoE or FairScale-MoE frameworks.
# Example:
#   moe = MoE(input_size=64, num_experts=8, hidden_size=128)
# =============================================================================

import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
class Expert(nn.Module):
    """
    Expert network class. Using Tanh as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - hidden_size (int): The size of the hidden layer.
    """
    def __init__(self,input_size,hidden_size):
        super(Expert, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(input_size,hidden_size), #[I,H]
            nn.Tanh(),
            nn.Linear(hidden_size,hidden_size) #[H,H]
        )
        
    def forward(self,x):
        return self.net(x) #[H,]


class Gating(nn.Module):
    """
    Gating network class.Using Relu as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - num_experts (int): The number of experts.
    - noise_epsilon (float): The noise epsilon value. default is 1e-4.
    """
    def __init__(self,input_size,num_experts,noise_epsilon=1e-4):
        super(Gating, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(input_size,num_experts) #[I,H]
        )
        self.noisy=nn.Linear(input_size,num_experts)
        self.softplus = nn.Softplus()
        self.noise_epsilon=noise_epsilon
    def forward(self,x,train):
        """ 
        - train (bool): Whether to train the model. Only add the noise when training.
        """
        gates=self.net(x)
        if train:
            noisy=self.softplus(self.noisy(x)) + self.noise_epsilon 
            std = torch.randn_like(gates)  
            output = gates + noisy * std
        else:
            output = gates
        
        return output#[E,]
        
    
class MoE(nn.Module):
    """MOE Block 

    Parameters:
    - input_size (int): The size of the input layer.
    - num_experts (int): The number of experts.
    - hidden_size (int): The size of the hidden layer.
    """
    def __init__(self,input_size,num_experts,hidden_size):
        super(MoE, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.experts = nn.ModuleList(
            [Expert(self.input_size,self.hidden_size) for _ in range(num_experts)]
            )
        
        self.softmax = nn.Softmax(dim=-1)
        self.gating_network = Gating(self.input_size,self.num_experts)
        
    def _gates_to_load(self, gates):
            """Compute the true load per expert, given the gates.
            The load is the number of examples for which the corresponding gate is >0.
            Args:
            gates: a `Tensor` of shape [E,]
            Returns:
            a float32 `Tensor` of shape [E,]
            """
            pass
    def topkGating(self,x,train):
            
            
            Gating = self.softmax(self.gating_network(x,train))
            values, indices= torch.topk(Gating,k=2,dim=-1) #values: [k,] indices: [k,]
            top_k_gates = values / (values.sum(0, keepdim=True) + 1e-6)  # normalization
            zeros = torch.zeros_like(Gating, requires_grad=True)
            gates = zeros.scatter(0, indices, top_k_gates) #Gating: [E,]
            #todo !!!
            #balance loss
            
            return gates
        
        
    def forward(self,x,train):
        gates= self.topkGating(x,train) #[E,]
        idx_gates = (gates != 0).float() #[E,]
        experts_outputs = [ self.experts[i](x) if idx_gates[i] != 0 else torch.zeros(self.hidden_size)
                            for i in range(self.num_experts)]#[E,H] 
        experts_outputs = torch.stack(experts_outputs,0)
        # print("expert_outputs-shape:",experts_outputs.shape)
        # print("gates-shape:",gates.shape)
        #todo!! balance loss
        
        output = torch.sum( gates.unsqueeze(1)* experts_outputs,dim=0) # mul along the first dimension -> [H,]
        return output



class MOE_Model(nn.Module):
    def __init__(self, input_size, num_experts, hidden_size,depth,output_size):
        super(MOE_Model, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size
        self.experts = nn.ModuleList([nn.Linear(input_size, hidden_size) for _ in range(num_experts)])
        self.gate = nn.Linear()
    ###todo!! vmap the model