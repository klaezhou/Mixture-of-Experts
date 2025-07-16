import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
from functorch import make_functional, vmap
import logging
import torch.nn.init as init
from  .. import moe

def get_network(args):
    pass

class MoE_FCNN(nn.Module):
    def __init__(self, input_size, num_experts, hidden_size, depth, output_size,k=2,loss_coef=1e-2,activation=nn.Tanh()):
        super(self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size
        self.k=k
        self.loss_coef=loss_coef
        self.moe=MoE(input_size, num_experts, hidden_size,self.k,self.loss_coef,activation)
        self.model = nn.ModuleList(
            [self.moe] +
            [MLP(hidden_size,activation) for _ in range(depth - 1)] +
            [nn.Linear(hidden_size, output_size)]
        )
        self._init_weights()
    def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)  # Xavier 正态分布初始化
                    if m.bias is not None:
                        init.zeros_(m.bias)
    def forward(self, x):
        loss=None
        for i, layer in enumerate(self.model):
            if i == 0:  # MoE 层需要 train 参数
                x,loss = layer(x, self.training)
            else:
                x = layer(x)
        return x,loss
    