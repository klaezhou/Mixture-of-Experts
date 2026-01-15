import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
from functorch import make_functional, vmap
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
        layer_list.append(nn.Linear(hidden_size,output_size))  #[H,I] zhou's model
        self.net = nn.ModuleList(layer_list)
        self._init_weights()
        
    
    def _init_weights(self):
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)  # Xavier 正态分布初始化
                    if m.bias is not None:
                        init.zeros_(m.bias)
    def forward(self, y):
        for i, layer in enumerate(self.net[:-1]):
            y = layer(y)
            y = self.activation(y)
        y = self.net[-1](y)
        # y = self.activation(y) #zhou's model
        return y
    

class GatingNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,depth=1,output_size=1,activation=nn.Tanh()):
        self.depth=depth
        super(GatingNetwork, self).__init__()
        self.activation=activation
        layer_list = []
        layer_list.append(nn.Linear(input_size,hidden_size))  #[I,H]
        for i in range(self.depth-1):
            layer_list.append(nn.Linear(hidden_size,hidden_size))  #[H,H]
        layer_list.append(nn.Linear(hidden_size,output_size,bias=False))  #[H,I] zhou's model
        self.net = nn.ModuleList(layer_list)
        self.softmax_eps=nn.Parameter(torch.tensor(0.0))
        self.softmax_eps.requires_grad=True
        self._init_weights()
        
    
    def _init_weights(self):
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)  # Xavier 正态分布初始化
                    if m.bias is not None:
                        init.zeros_(m.bias)
    def forward(self, y):
        for i, layer in enumerate(self.net[:-1]):
            y = layer(y)
            y = self.activation(y)
        y = self.net[-1](y)
        y= F.sigmoid(y/torch.exp(self.softmax_eps))
        return y
    
    
class PF_moe(nn.Module):
    def __init__(self,input_size,action_size,hidden_size,expert_depth,gating_hiddem_size,gating_depth=1,num_experts=2):
        super(PF_moe, self).__init__()
        self.num_experts=num_experts
        self.experts=nn.ModuleList([Expert(input_size,hidden_size,action_size,expert_depth) for _ in range(num_experts)])
        self.gating_network=GatingNetwork(input_size,gating_hiddem_size,gating_depth)
        self._report_trainable()

    def _report_trainable(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"=== Trainable parameters ===\nTotal trainable params: {total}\n")
        
    def forward(self,obs):
        gating_weights=self.gating_network(obs)  # [B,num_experts]
        expert_output1=self.experts[0](obs)  # [B,action_size]
        expert_output2=self.experts[1](obs)  # [B,action_size]
        action=gating_weights*expert_output1+(1-gating_weights)*expert_output2
         # [B,action_size]
        return action
    
    
class Conv_nn(nn.Module):
    def __init__(self,input_size,kernel_size, stride,out_channels=1):
        super(Conv_nn, self).__init__()
        self.padding=(kernel_size-1)//2
        self.kernel_size=kernel_size
        self.stride=stride
        self.conv_layers = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, 
                                     stride=stride, padding=self.padding) # outputsize = L_out = floor((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        self.input_size=input_size
        self._init_weights()
        self.conv_layers.requires_grad_(False)
    def _init_weights(self):
            # 遍历模型中的所有模块
            for m in self.modules():
                # 注意：isinstance 是一个单词，中间没有空格
                if isinstance(m, nn.Conv1d):
                    # 获取卷积核大小，计算初始值
                    #——————————————————————————————————————————————————————————
                    # m.kernel_size 是一个元组，例如 (3,)，所以取 m.kernel_size[0]
                    val = 1.0 / m.kernel_size[0]
                    
                    # 将权重全部初始化为固定值 1/k
                    init.constant_(m.weight, val)
                    
                    #————————————————————————————————————————————————————————————————————————————
                    # k = m.kernel_size[0]
                    # # 1. 定义高斯核的标准差 sigma (通常设为 k/6 左右比较合适)
                    # sigma = k / 8.0 
                    # center = (k - 1) / 2.0
                    
                    # # 2. 计算一维高斯序列
                    # x = torch.arange(k).float()
                    # # 高斯公式：exp(-(x - center)^2 / (2 * sigma^2))
                    # gauss = torch.exp(-(x - center)**2 / (2 * sigma**2))
                    
                    # # 3. 归一化 (可选)：让所有权值的和为 1，这样不会改变信号的总强度
                    # gauss = gauss / gauss.sum()
                    
                    # # 4. 将计算好的高斯值赋给权重
                    # # m.weight 的形状是 (out_channels, in_channels, kernel_size)
                    # # 我们需要将计算好的一维核广播到所有通道
                    # with torch.no_grad():
                    #     m.weight.copy_(gauss.view(1, 1, k).expand_as(m.weight))
                    
                    #————————————————————————————————————————————————————————————————————————————
    def get_input(self,y):
        l_out = (self.input_size + 2*self.padding - (self.kernel_size - 1) - 1) // self.stride + 1

        # 记录每个输出位点对应的输入 index 中心
        indices = []
        for i in range(l_out):
            # 计算逻辑：i 是输出的索引，我们要找它对应输入的中心
            # 中心索引 = i * stride - padding + (kernel_size - 1) / 2
            center_index = i * self.stride - self.padding + (self.kernel_size - 1) / 2
            indices.append(center_index)
        y_new=y[:,indices]
        return y_new
        
    def forward(self, y):
        y = self.conv_layers(y)
        return y