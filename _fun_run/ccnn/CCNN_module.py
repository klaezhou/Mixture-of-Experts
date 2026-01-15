import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
from functorch import make_functional, vmap
import logging
import torch.nn.init as init
from moe_module.moe import Gating


class CCNN(nn.Module):
    def __init__(self, input_size, num_experts, wide_size, out_channel, output_size,k=1,activation=nn.Tanh()):
        super(CCNN, self).__init__()
        self.convolution = nn.Conv1d(in_channels=input_size, out_channels=1, kernel_size=k, padding=k//2)
        self.gating_network = Gating(input_size,num_experts)
        
