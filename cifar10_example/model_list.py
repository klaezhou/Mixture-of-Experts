import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
from functorch import make_functional, vmap
import logging
import torch.nn.init as init
from  moe_module.utils import get_activation,get_loss_fn,get_optimizer,log_with_time,plot_dual_axis
from moe_module.moe import SparseDispatcher,Expert,Gating,MoE,MLP

def get_network(args):
    if args.model=="moe_res_fcnn":
        return MoE_Res_FCNN(args.input_size, args.num_experts,args.hidden_size,args.depth, args.output_size,args.k,args.loss_coef,get_activation(args.activation))
    if args.model=="udi_res_fcnn":
        return UDI_Res_FCNN(args.input_size, args.hidden_size,args.depth, args.output_size,get_activation(args.activation))




class MoE_Res_FCNN(nn.Module):
    # 用简单的flatten +linear 层来处理图像
    def __init__(self, input_size, num_experts, hidden_size, depth, output_size,k=2,loss_coef=1e-2,activation=nn.Tanh()):
        super(MoE_Res_FCNN,self).__init__()
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
            [ResMLP(hidden_size,activation) for _ in range(depth - 1)] +
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


class UDI_Res_FCNN(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size,activation=nn.Tanh()):
        super(UDI_Res_FCNN,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size
        self.udi = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.udi_init(self.udi)
        self.layer1=nn.Sequential(self.udi, activation)
        self.model = nn.ModuleList(
            [self.layer1]+
            [ResMLP(hidden_size,activation) for _ in range(depth )] +
            [nn.Linear(hidden_size, output_size)]
        )
        self._init_weights()

    def udi_init(self,linear_layer, R=1.0, gamma=1.0):
        """
        UDI 初始化 Linear 层
        :param linear_layer: nn.Linear 层
        :param R: 偏置 b_j 的取值范围 [0, R]
        :param gamma: 缩放系数（用于 tanh 前）
        """
        with torch.no_grad():
            # a_j: 均匀地分布在单位球面
            weight = torch.randn_like(linear_layer.weight)  # shape [out_features, in_features]
            weight = weight / weight.norm(dim=1, keepdim=True)  # 每一行归一化
            linear_layer.weight.copy_(gamma * weight)

            # b_j: 从 [0, R] 中均匀采样
            bias = torch.empty_like(linear_layer.bias)
            bias.uniform_(0, R)
            linear_layer.bias.copy_(bias)
    def _init_weights(self):
        for m in self.modules():
            # 如果 m 是 self.udi 或它的子模块，就跳过
            # if any(m is sub for sub in self.udi.modules()):
            #     continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    

    def forward(self, x):
        for i, layer in enumerate(self.model):
                x = layer(x)
        return x

    
    
class MOE_CNN(nn.Module):
     pass 
    
"""        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Flatten(),     # 64 * 8 * 8 = 4096
            nn.Linear(64 * 8 * 8, hidden_size)
        )"""
    
class ResMLP(nn.Module):
    def __init__(self, hidden_size, activation):
        super(ResMLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = activation
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        res=self.act(out + identity)
        return   res# 残差连接

