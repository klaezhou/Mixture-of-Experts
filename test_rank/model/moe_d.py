# =============================================================================
# Mixture of Experts Module for Function Approximation

# moe with dense gates, 

# =============================================================================

import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
import logging
import torch.nn.init as init


    
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
        # for layer in self.net:
        #     self.udi_init(layer)
        self._report_trainable()
        
    def _report_trainable(self):
            total = 0
            print("=== MOE Trainable parameters ===")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    n = p.numel()
                    total += n
            print(f"MOE trainable params: {total}")
        
    def udi_init(self, layer, gamma=1, R=1):
            # 取出层参数
            weight = torch.randn_like(layer.weight)  # 随机方向
            # 每一行归一化到单位球面（a_j）
            weight = F.normalize(weight, p=2, dim=1)
            # 可选缩放 γ
            layer.weight.data = gamma * weight
            # 偏置均匀分布在 [0, R]
            layer.bias.data.uniform_(0.0, R)
            layer.udi_initialized = True
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


class Gating(nn.Module):
    """
    Gating network class.Using Relu as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - num_experts (int): The number of experts.
    - noise_epsilon (float): The noise epsilon value. default is 1e-4.
    """
    def __init__(self,input_size,num_experts,gating_hidden_size,gating_depth,activation=nn.Tanh(),gamma=1,R=2):
        super(Gating, self).__init__()
        self.activation=activation
        layer_list = []
        layer_list.append(nn.Linear(input_size,gating_hidden_size))  #[I,H]
        for i in range(gating_depth-1):
            layer_list.append(nn.Linear(gating_hidden_size,gating_hidden_size))  #[H,H]
        layer_list.append(nn.Linear(gating_hidden_size,num_experts))  #[H,E] 
        self.net = nn.ModuleList(layer_list)
        self.softmax = nn.Softmax(dim=-1)
        # self.udi_init(self.net[0], gamma, R)
    def udi_init(self, layer, gamma, R):
            # 取出层参数
            weight = torch.randn_like(layer.weight)  # 随机方向
            # 每一行归一化到单位球面（a_j）
            weight = F.normalize(weight, p=2, dim=1)
            # 可选缩放 γ
            layer.weight.data = gamma * weight
            # 偏置均匀分布在 [0, R]
            layer.bias.data.uniform_(0.0, R)
            layer.udi_initialized = True
    
    @torch.no_grad()
    def forward_int(self, y):
        #for integration
        for i, layer in enumerate(self.net[:-1]):
            y = layer(y)
            y = self.activation(y)
        y = self.net[-1](y)
        y=torch.sin(5*torch.pi*y)
        # y= self.softmax(y)
        return y
    
    def forward(self,y):
        for i, layer in enumerate(self.net[:-1]):
            y = layer(y)
            y = self.activation(y)
        
        y = self.net[-1](y)
        y=torch.sin(5*torch.pi*y)
        # y= self.softmax(y)
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
        
        self.gating_network = Gating(self.input_size,self.num_experts,self.gating_hidden_size,self.gating_depth)
    

    def forward(self,x):
        gates=self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # Shape: (batch_size, num_experts, output_size)
        
        self.gates_check=gates.detach()
        y = (gates.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (B, out)


        return y



class MOE_dense(nn.Module):   
    def __init__(self, args):
        super().__init__()

        def _get(name, default):
            return getattr(args, name, default)

        # ---- infer sizes (very handy for PDE) ----
        # For 2D Poisson: input_size=2, output_size=1
        self.input_size  = _get("input_size", 2)
        self.output_size = _get("output_size", 1)

        # ---- MoE config with defaults ----
        self.num_experts = _get("num_experts", 4)
        self.hidden_size = _get("hidden_size", 20)
        self.depth       = _get("depth", 3)

        self.gating_hidden_size = _get("gating_hidden_size", 10)
        self.gating_depth       = _get("gating_depth", 2)

        self.activation = nn.Tanh()

        self.moe = MoE(
            input_size=self.input_size,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            depth=self.depth,
            output_size=self.output_size,
            gating_hidden_size=self.gating_hidden_size,
            gating_depth=self.gating_depth,
            activation=self.activation
        )

        self.model = self.moe

        self._init_weights()
        self._report_trainable()
        
    def _report_trainable(self):
            total = 0
            print("=== Trainable parameters ===")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    n = p.numel()
                    total += n
            print(f"Total trainable params: {total}")
        
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
    



@torch.no_grad()
def moe_penultimate_getter(model, x):
    """
    Feature getter
    This function are used for model is a MOE_dense instance.
    Phi = concat over experts of penultimate hidden activations.
    Returns: Phi [N, num_experts * hidden_size]
    """
    moe = model.model  # MOE_dense: self.model = self.moe
    feats = []
    g= moe.gating_network.forward_int(x)  # [N, E]
    for e, expert in enumerate(moe.experts):
        h = expert.forward_penultimate(x)           # [N, H]
        h = h * g[:, e:e+1]                         # [N, H], broadcast weight
        feats.append(h)

    Phi = torch.cat(feats, dim=1)          # [N, E*H]
    return Phi