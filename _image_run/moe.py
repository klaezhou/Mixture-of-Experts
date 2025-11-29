


import torch
import torch.nn as nn
import numpy as np 
from torch.distributions.normal import Normal
import torch.nn.functional as F
from functorch import make_functional, vmap
import logging
import torch.nn.init as init
from _model import BasicBlock, ResNet20, LeNet, Gate_net_CIFAR10


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
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




        
    
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
        
        self.Resnet1=ResNet20()
        Path1= f"/home/zhy/Zhou/mixture_of_experts/_image_run/saved_cnn/_f5cifar10_cnn.pt"
        self.Resnet1.load_state_dict(torch.load(Path1, weights_only=True))
        self.Resnet2=ResNet20()
        Path2= f"/home/zhy/Zhou/mixture_of_experts/_image_run/saved_cnn/_l5cifar10_cnn.pt"
        self.Resnet2.load_state_dict(torch.load(Path2, weights_only=True))
        self.experts = nn.ModuleList(
            [self.Resnet1, self.Resnet2]
            )
        
        self.experts.requires_grad_(False)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gating_network = Gate_net_CIFAR10()
        self.tau1 ,self.tau2=torch.tensor(1e-5,requires_grad=True),torch.tensor(1e-5,requires_grad=True)

    def smoothing(self,step,step_lb):
        # cross train  -> soft
        self.smooth = not self.smooth
        if step >=step_lb:
            self.smooth =False

        
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

    def soft_topk(self,s: torch.Tensor,k: int): #,tau1: float = 5e-2,tau2: float = 1e-2
        
        s=F.softmax(s,dim=-1)
        diff = torch.unsqueeze(s,-1) - torch.unsqueeze(s,-2)
        sigma = torch.sigmoid(-diff / self.tau1)
        row_sum = sigma.sum(dim=-1) - 0.5
        r_tilde = 1.0 + row_sum
        eps=0.5    
        a = torch.sigmoid((k+eps - r_tilde) / self.tau2)     

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
    

    
class MOE_LeNET_ResNet(nn.Module):   
    def __init__(self, input_size, num_experts, hidden_size, depth, output_size,k=2,loss_coef=1e-2,activation=nn.Tanh()):
        super(MOE_LeNET_ResNet, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.depth = depth
        self.output_size = output_size
        self.k=k
        self.loss_coef=loss_coef
        self.moe=MoE(input_size, num_experts, hidden_size,depth,output_size,self.k,self.loss_coef,activation)
        self.model=self.moe


        
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
        
        output,loss1=self.model(x, self.training)
        # beta=self.Beta(x)
        # output=(output*beta).sum(dim=1,keepdim=True)
        # output=self.linear(output)
        loss=loss1
        return output,loss
    