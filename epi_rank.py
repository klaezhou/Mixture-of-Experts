import torch
import torch.nn as nn
import numpy as np
#numerical integral to compute epi_rank
class epi_rank_moe():
    def __init__(self, model,interval,num_samples):
        """
        model: class MOE_Model in moe.py
        """
        self.model = model
        
        #find the MOE in MOE_Model
        # moe example: y,loss=self.moe(x,train)
        self.moe = model.moe
        self.device=next(model.parameters()).device
        self.epsilon =1e-6
        interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
        self.x = torch.linspace(interval[0], interval[1], num_samples).view(-1, 1).to(self.device)
        self.num_samples = num_samples
        self.training=False
        self.M_weight=self.compute_matrix()
        self.rank=0
        
        
    def compute_matrix(self):
        """moe 
        input [Batch,1,] -> Gating [Batch,num_experts] * 
        Experts [Batch,1,hidden_size] ->
        """
        d_matrix,loss=self.moe(self.x,self.training)
        d_matrix=d_matrix.detach() #[Batch,hidden_size]
        #init weight matrix, numerical
        def diag_weight(size):
            #梯形公式
            weights = torch.ones(size)
            weights[0] = weights[-1] = 0.5
            diag_matrix = torch.diag(weights).to(self.device)
            return diag_matrix
        weight_matrix=diag_weight(self.num_samples)
        #D.T W D
        M_weight = torch.matmul(d_matrix.t(), weight_matrix)
        M_weight = torch.matmul(M_weight, d_matrix)
        # y_np = d_matrix.cpu().numpy()
        # print("moe ouput:", y_np.shape)
        self.M_weight = M_weight
        return  M_weight
        
    def rank_moe(self):
        eigvals = torch.linalg.eigvalsh(self.M_weight)

        # 设定阈值 epsilon
        epsilon = 1e-6

        # 统计大于 epsilon 的特征值数量
        count = (eigvals > epsilon).sum().item()
        self.rank = count
        return count
        
        
class epi_rank_mlp():
    def __init__(self, model,interval,num_samples,moe_training=True,index=2):
        """index: defalut 2 ; if no moe, then 1"""
        self.moe_training=moe_training
        self.model = model
        self.index=index
        #find the MLP in MOE_Model
        self.mlp = [ PartialMOE(model,n,self.index) for n in range(len(model.model)-2)]
        self.device=next(model.parameters()).device
        self.epsilon =1e-6
        interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
        self.x = torch.linspace(interval[0], interval[1], num_samples).view(-1, 1).to(self.device)
        self.num_samples = num_samples
        self.training=False
        self.M_weight_list=self.compute_matrix_list()
        self.rank_list=0
    def compute_matrix_list(self):
                d_matrix_list=[]
                for i in range(len(self.mlp)):
                    d_matrix=self.mlp[i](self.x,self.training,self.moe_training)
                    d_matrix=d_matrix.detach()
                    d_matrix_list.append(d_matrix)

                #init weight matrix, numerical
                def diag_weight(size):
                    #梯形公式
                    weights = torch.ones(size)
                    weights[0] = weights[-1] = 0.5
                    diag_matrix = torch.diag(weights).to(self.device)
                    return diag_matrix
                weight_matrix=diag_weight(self.num_samples)
                #D.T W D
                M_weight_list=[]
                for i in range(len(d_matrix_list)):
                    M_weight = torch.matmul(d_matrix_list[i].t(), weight_matrix)
                    M_weight = torch.matmul(M_weight, d_matrix_list[i])
                    M_weight_list.append(M_weight)
                self.M_weight_list = M_weight_list
                # y_np = d_matrix.cpu().numpy()
                # print("moe ouput:", y_np.shape)
                return  M_weight_list
        
    def rank_mlp(self):
        rank_list=[]
        for i in range(len(self.M_weight_list)):
            eigvals = torch.linalg.eigvalsh(self.M_weight_list[i])

            # 设定阈值 epsilon
            epsilon = 1e-6

            # 统计大于 epsilon 的特征值数量
            count = (eigvals > epsilon).sum().item()
            rank_list.append(count)
        self.rank_list = rank_list
        return rank_list
    
class PartialMOE(nn.Module):
    def __init__(self, model, n,index=2):
        super().__init__()
        self.model = model
        self.n = n
        self.index=index

    def forward(self, x,training,moe_traing=True):
        for i, layer in enumerate(self.model.model[:self.n+self.index]):
            if i == 0 and moe_traing:
                x, _ = layer(x,training)
            else:
                x = layer(x)
        return x
    