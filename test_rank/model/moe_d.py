# =============================================================================
# Mixture of Experts Module for Functional Approximation

# moe with dense gates, 

# =============================================================================

import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
import logging
import torch.nn.init as init

torch.set_default_dtype(torch.float64)

    
class Expert(nn.Module):
    """
    Expert network class. Using Tanh as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - hidden_size (int): The size of the hidden layer.
    """
    def __init__(self,input_size,hidden_size,output_size,depth,activation=nn.Tanh()):
        self.depth=depth
        super(Expert, self).__init__()
        self.activation=activation
        layer_list = []
        layer_list.append(nn.Linear(input_size,hidden_size))  #[I,H]
        for i in range(self.depth-1):
            layer_list.append(nn.Linear(hidden_size,hidden_size))  #[H,H]
        layer_list.append(nn.Linear(hidden_size,output_size))  #[H,I] 
        self.net = nn.ModuleList(layer_list)
    
    def forward(self, y):
        for i, layer in enumerate(self.net[:-1]):
            y = layer(y)
            y = self.activation(y)
        y = self.net[-1](y)
        return y


class Gating(nn.Module):
    """
    Gating network class.Using Relu as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - num_experts (int): The number of experts.
    - noise_epsilon (float): The noise epsilon value. default is 1e-4.
    """
    def __init__(self,input_size,num_experts,gating_hidden_size,gating_depth,activation=nn.Tanh(),gamma=2,R=2):
        super(Gating, self).__init__()
        self.activation=activation
        layer_list = []
        layer_list.append(nn.Linear(input_size,gating_hidden_size))  #[I,H]
        for i in range(gating_depth-1):
            layer_list.append(nn.Linear(gating_hidden_size,gating_hidden_size))  #[H,H]
        layer_list.append(nn.Linear(gating_hidden_size,num_experts))  #[H,E] 
        self.net = nn.ModuleList(layer_list)
                
        # self.udi_init(self.net[0], gamma, R)
        # self.net[0].udi_initialized = True
    def udi_init(self, layer, gamma, R):
            # 取出层参数
            weight = torch.randn_like(layer.weight)  # 随机方向
            # 每一行归一化到单位球面（a_j）
            weight = F.normalize(weight, p=2, dim=1)
            # 可选缩放 γ
            layer.weight.data = gamma * weight
            # 偏置均匀分布在 [0, R]
            layer.bias.data.uniform_(0.0, R)

    def forward(self,y):
        for i, layer in enumerate(self.net[:-1]):
            y = layer(y)
            y = self.activation(y)
        y = self.net[-1](y)

        
        return y
        
    
class MoE(nn.Module):
    """MOE Block 

    Parameters:
    - input_size (int): The size of the input layer.
    - num_experts (int): The number of experts.
    - hidden_size (int): The size of the hidden layer.
    """
    def __init__(self,input_size,num_experts,hidden_size,depth,output_size,gating_hidden_size,gating_depth,activation=nn.Tanh()):
        super(MoE, self).__init__()
        torch.set_default_dtype(torch.float64)
        self.depth = depth
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gating_hidden_size = gating_hidden_size
        self.gating_depth = gating_depth
        self.experts = nn.ModuleList(
            [Expert(self.input_size,self.hidden_size,self.output_size,self.depth,activation) for _ in range(num_experts)]
            )
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.gates_check=None
        self.softmax = nn.Softmax(dim=-1)
        self.gating_network = Gating(self.input_size,self.num_experts,self.gating_hidden_size,self.gating_depth)
    
        self.report_trainable()
    def report_trainable(self):
            total = 0
            print("=== MoE Trainable parameters ===")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    n = p.numel()
                    total += n
            print(f"Total trainable params: {total}\n")

    def forward(self,x):
        gates=self.gating_network(x)
        
        gates= self.softmax(gates)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: (batch_size, num_experts, output_size)
        
        self.gates_check=gates
        gates=gates.unsqueeze(-1)  # Shape: (batch_size, num_experts, 1)
        y=gates*expert_outputs

        return y

class MLP(nn.Module):
    def __init__(self, hidden_size,activation=nn.Tanh()):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_size),
            activation
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)  # Xavier 正态分布初始化
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = self.model(x)
        return x

    
class MOE_dense(nn.Module):   
    def __init__(self,args):
        super(MOE_dense, self).__init__()
        self.input_size = args.input_size
        self.num_experts = args.num_experts
        self.hidden_size = args.hidden_size
        self.depth = args.depth
        self.output_size = args.output_size
        self.activation=nn.Tanh()
            # === 1️⃣ MoE 网络 ===
        self.moe=MoE(input_size=self.input_size,
                     num_experts=self.num_experts,
                     hidden_size=self.hidden_size,
                     depth=self.depth,
                     output_size=self.output_size,
                    gating_hidden_size=args.gating_hidden_size,
                    gating_depth=args.gating_depth,
                    activation=self.activation)
        self.model=self.moe
        
        self._init_weights()
        self._report_trainable()
        
        
    def _report_trainable(self):
            total = 0
            print("=== Trainable parameters ===")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    n = p.numel()
                    total += n
            print(f"Total trainable params: {total}\n")
        
    def _init_weights(self):
        import torch.nn.init as init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 如果是 UDI 初始化过的层，就跳过！
                if getattr(m, "udi_initialized", False):
                    continue
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(self, x):
        output=self.model(x)
        return output