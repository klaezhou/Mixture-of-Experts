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
from torch.distributions.normal import Normal
import torch.nn.functional as F
from functorch import make_functional, vmap
import logging
import torch.nn.init as init


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index] 
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.double())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
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


class Gating(nn.Module):
    """
    Gating network class.Using Relu as activation function.

    Parameters:
    - input_size (int): The size of the input layer.
    - num_experts (int): The number of experts.
    - noise_epsilon (float): The noise epsilon value. default is 1e-4.
    """
    def __init__(self,input_size,num_experts,noise_epsilon=1e-6,gamma=1,R=1):
        super(Gating, self).__init__()
        self.net=nn.Sequential(
            # nn.Linear(input_size,num_experts)
            nn.Linear(input_size,num_experts*1),
            # nn.Tanh(),
            # nn.Linear(num_experts*5,num_experts*5),
            nn.Tanh(),
            nn.Linear(num_experts*1,num_experts),
            #[I,H]
        )
        self.noisy=nn.Linear(input_size,num_experts)
        self.softplus = nn.Softplus()
        self.noise_epsilon=noise_epsilon
        
        

                
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
            

    def forward(self,x,train):
        """ 
        - train (bool): Whether to train the model. Only add the noise when training.
        """
        gates=self.net(x)
        noisy_stddev=None 
        # if train:
        #     noisy_stddev=self.softplus(self.noisy(x)) + self.noise_epsilon 
        #     std = torch.randn_like(gates)  
        #     output= gates + noisy_stddev * std
        # else:
        #     output = gates
        output=gates
        
        return output, gates,noisy_stddev#[E,] noisy - clean-nosiy_stddev
        
    
class MoE(nn.Module):
    """MOE Block 

    Parameters:
    - input_size (int): The size of the input layer.
    - num_experts (int): The number of experts.
    - hidden_size (int): The size of the hidden layer.
    """
    def __init__(self,input_size,num_experts,hidden_size,depth,output_size,k=2,loss_coef=1e-2,activation=nn.Tanh(),epsilon=1e-0):
        super(MoE, self).__init__()
        self.k=k
        self.depth = depth
        self.smooth=True
        self.loss_coef = loss_coef
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epsilon = epsilon
        self.experts = nn.ModuleList(
            [Expert(self.input_size,self.hidden_size,self.output_size,self.depth,activation) for _ in range(num_experts)]
            )
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.gates_check=None
        self.softmax = nn.Softmax(dim=-1)
        self.gating_network = Gating(self.input_size,self.num_experts)
    
    def smoothing(self,step,step_lb):
        self.smooth = not self.smooth
        if step >=step_lb:
            self.smooth = True
        # for p in self.gating_network.parameters():
        #     p.requires_grad = self.smooth
        
    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
        normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
        "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k            #[batch]
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)   #[batch,1]
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob #[batch,num_experts]

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.var() / (x.mean()**2 + eps)
    

    
    def topkGating(self,x,train):
            ## topk--> softmax
            noisy,clean,noisy_stddev=self.gating_network(x,train)
            values=noisy
            
            values=self.softmax(values)
            values, indices= torch.topk(values,k=self.k,dim=-1) 
            # top_logits,_=torch.topk(noisy,k=self.k+1,dim=-1) #values: [k,] indices: [k,] zhou's mode
            # values=values/(values.sum(1, keepdim=True) + 1e-8)  # normalization
            # values= values / (values.sum(1, keepdim=True) + 1e-8) 
            zeros= torch.zeros_like(noisy, requires_grad=True)
            gates=zeros.scatter(1, indices, values)
            ## softmax--> topk-->normalize
            # Gating = self.softmax(noisy)
            # values, indices= torch.topk(Gating,k=self.k,dim=-1) 
            # top_logits,_=torch.topk(Gating,k=self.k+1,dim=-1) #values: [k,] indices: [k,]
            # top_k_gates = values / (values.sum(1, keepdim=True) + 1e-8)  # normalization
            # zeros = torch.zeros_like(Gating, requires_grad=True)
            # gates = zeros.scatter(1, indices, top_k_gates)#Gating: [E,]
            #balance loss
            # if  train:
            #     load = (self._prob_in_top_k(clean, noisy, noisy_stddev, top_logits)).sum(0) #[num_experts,]
            # else:
            #     load = self._gates_to_load(gates)
            #     load=load.float()   #zhou'model
            load=0
                
            
            return gates,load
        
    def soft_topk(self,s: torch.Tensor,k: int,tau1: float = 5e-2,tau2: float = 1e-2):
        s=F.softmax(s,dim=-1)
        diff = torch.unsqueeze(s,-1) - torch.unsqueeze(s,-2)
        sigma = torch.sigmoid(-diff / tau1)
        row_sum = sigma.sum(dim=-1) - 0.5
        r_tilde = 1.0 + row_sum
        eps=0.5    
        a = torch.sigmoid((k+eps - r_tilde) / tau2)     

        a=a*s
        # a=a/(a.sum(1, keepdim=True) + 1e-8)
        

        return a
    def forward(self,x,train):
        smooth=self.smooth
        if smooth:
            
            gates,_,_=self.gating_network(x,train)
            gates= self.soft_topk(gates, self.k)
            # gates= gates/(self.epsilon) # +1e-1*torch.abs(gates)
            # gates= self.softmax(gates)
            # gates= gates / (gates.sum(1, keepdim=True) + 1e-8) 
        else:
            gates,load= self.topkGating(x,train)
        
        self.gates_check=gates
        # if not train:
        #     # not train-> print gates
        #     gates_np = gates.detach().cpu().numpy()
        #     print("Gates:\n", np.array2string(gates_np, precision=4, suppress_small=True))
        importance=gates.sum(0) 
        ##new dispatcher
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates=dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        #balance loss
        #batch wise
        loss_coef=self.loss_coef
        loss=self.cv_squared(importance) # + self.cv_squared(load) zhou'model
        loss*=loss_coef
        return y,loss

class MLP(nn.Module):
    def __init__(self, hidden_size,activation=nn.Tanh()):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, hidden_size),
            activation
        )

    def forward(self, x):
        x = self.model(x)
        return x



# example model class MoE_Model
    

class MLP_Model(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size, activation=nn.Tanh()):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.depth       = depth
        self.output_size = output_size

        assert input_size == 1, "当前归一化假设输入为 [x,t] 两维。"

        # 边界可改成参数传入，这里先保留你的写法
        self.register_buffer("lb", torch.tensor([-1.0]))
        self.register_buffer("ub", torch.tensor([ 1.0]))

        # 建层（建议每层各自一个激活实例）
        def make_act():
            return activation.__class__() if isinstance(activation, nn.Module) else activation

        layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), make_act())
        blocks = [layer1] + [MLP(hidden_size, make_act()) for _ in range(depth - 1)] + [nn.Linear(hidden_size, output_size)]
        self.model = nn.ModuleList(blocks)
        self.apply(self._init)
        self._report_trainable()
        

    def _report_trainable(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"=== Trainable parameters ===\nTotal trainable params: {total}\n")

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, normalize=True):
        # 保证 dtype/device 一致，防止外部是 float64/GPU 而 buffer 不一致
        if normalize:
            lb = self.lb.to(dtype=x.dtype, device=x.device)
            ub = self.ub.to(dtype=x.dtype, device=x.device)
            y = 2.0 * (x - lb) / (ub - lb) - 1.0
        else:
            y = x.to(dtype=self.lb.dtype, device=self.lb.device)

        for i, layer in enumerate(self.model):
            y = layer(y)
        return y
    

    
class MOE_modify_beta(nn.Module):   
    def __init__(self, input_size, num_experts, hidden_size, depth, output_size,k=2,loss_coef=1e-2,activation=nn.Tanh()):
        super(MOE_modify_beta, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size
        self.k=k
        self.loss_coef=loss_coef
        self.moe=MoE(input_size, num_experts, hidden_size,depth,hidden_size,self.k,self.loss_coef,activation)
        self.model=self.moe
        lb=[-1,0]
        ub=[1,1]
        self.register_buffer("lb", torch.as_tensor(lb))
        self.register_buffer("ub", torch.as_tensor(ub))
        # === 2️⃣ Beta 网络（可调 depth） ===
        # layers = []
        # in_dim = input_size
        # # 如果 depth = 1，则只有一层线性映射
        # for i in range(1):
        #     layers.append(nn.Linear(in_dim, hidden_size))
        #     layers.append(nn.ReLU())
        #     in_dim = hidden_size

        # layers.append(nn.Linear(hidden_size, hidden_size,bias=False))
        # self.Beta = nn.Sequential(*layers)
        
        self.linear=nn.Linear(hidden_size, output_size)
        

        
        self._init_weights()
        self._report_trainable()
        
    def frozen_beta(self):
            for p in self.Beta.parameters():
                p.requires_grad_(False)
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
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        output,loss1=self.model(x, self.training)
        # beta=self.Beta(x)
        # output=(output*beta).sum(dim=1,keepdim=True)
        output=self.linear(output)
        loss=loss1
        return output,loss
    
    

    
class MOE_modify_betap(nn.Module):   
    def __init__(self, input_size, num_experts, hidden_size, depth, output_size,k=2,loss_coef=1e-2,activation=nn.Tanh()):
        super(MOE_modify_betap, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size
        self.k=k
        self.loss_coef=loss_coef
        self.moe=MoE(input_size, num_experts, hidden_size,depth,output_size,self.k,self.loss_coef,activation)
        self.model=self.moe
        lb=[-1]
        ub=[1]
        self.register_buffer("lb", torch.as_tensor(lb))
        self.register_buffer("ub", torch.as_tensor(ub))
        # === 2️⃣ Beta 网络（可调 depth） ===
        # layers = []
        # in_dim = input_size
        # # 如果 depth = 1，则只有一层线性映射
        # for i in range(1):
        #     layers.append(nn.Linear(in_dim, hidden_size))
        #     layers.append(nn.ReLU())
        #     in_dim = hidden_size

        # layers.append(nn.Linear(hidden_size, hidden_size,bias=False))
        # self.Beta = nn.Sequential(*layers)
        
        # self.linear=nn.Linear(hidden_size, output_size)
        

        
        self._init_weights()
        self._report_trainable()
        
    def frozen_beta(self):
            for p in self.Beta.parameters():
                p.requires_grad_(False)
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
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        output,loss1=self.model(x, self.training)
        # beta=self.Beta(x)
        # output=(output*beta).sum(dim=1,keepdim=True)
        # output=self.linear(output)
        loss=loss1
        return output,loss
    