import torch
import torch.nn as nn
import numpy as np

class epi_rank_mlp():
    def __init__(self, model,x_interval,t_interval,x_num_samples, t_num_samples,epsilon, moe_training=True,index=0):
        """index: defalut 0 ; if no moe, then 1"""
        self.moe_training=moe_training
        self.model = model
        self.index=index
        #find the MLP in MOE_Model moe_model => moe-*fcnn-ouputlayer, index=2 表示从第一个fcnn开始计算 
        #self.model.model[:self.n+self.index] 表示到前n+index个fcnn
        
        
        self.device=next(model.parameters()).device
        self.epsilon = epsilon

        x_interval = eval(x_interval)  # 例如 "[-1,1]" -> [-1, 1]
        self.x = torch.linspace(x_interval[0], x_interval[1], x_num_samples).view(-1, 1).to(self.device)
        self.x_num_samples = x_num_samples

        t_interval = eval(t_interval)
        self.t = torch.linspace(t_interval[0], t_interval[1], t_num_samples).view(-1, 1).to(self.device)
        self.t_num_samples = t_num_samples

        X, T = torch.meshgrid(self.x.squeeze(), self.t.squeeze(), indexing='ij')
        X_total = torch.stack([X.flatten(), T.flatten()], dim=1)
        self.X_total = X_total
        if moe_training:
            self.mlp = [ PartialMOE(model,n,self.index) for n in range(index)]
        else:
            self.mlp = [ PartialMOE(model,n,self.index) for n in range(len(model.model)-index)]
        self.training=False
        self.M_weight_list=self.compute_matrix_list()
        if moe_training:
            self.M_weight_list_experts=self.compute_experts_matrix_list()
        self.rank_list_experts=0
        self.rank_list=0

    def compute_matrix_list(self):
        d_matrix_list=[]
        for i in range(len(self.mlp)):
            d_matrix=self.mlp[i](self.X_total,self.training,self.moe_training)
            d_matrix=d_matrix.detach()
            d_matrix_list.append(d_matrix)

        #init weight matrix, numerical
        def diag_weight(x_size, t_size):
            #梯形公式
            weights = torch.ones((x_size, t_size))
            weights[0, :] *= 0.5
            weights[-1, :] *= 0.5
            weights[:, 0] *= 0.5
            weights[:, -1] *= 0.5
            weights = weights.flatten()
            diag_matrix = (self.x[-1]-self.x[0])/(x_size-1) * (self.t[-1]-self.t[0])/(t_size-1) * torch.diag(weights).to(self.device)
            return diag_matrix
        
        weight_matrix=diag_weight(self.x_num_samples, self.t_num_samples)
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
            # 统计大于 epsilon 的特征值数量
            count = (eigvals > self.epsilon).sum().item()
            rank_list.append(count)
        self.rank_list = rank_list
        return rank_list
    
    def compute_experts_matrix_list(self):
        d_matrix_list=[]
        for i in range(self.model.num_experts):
           
            partial_model=PartialExpert(self.model,i)
            d_matrix=partial_model(self.X_total)
            d_matrix=d_matrix.detach()
            d_matrix_list.append(d_matrix)

        big_matrix = torch.cat(d_matrix_list, dim=1)
        d_matrix_list.append(big_matrix)


        #init weight matrix, numerical
        def diag_weight(x_size, t_size):
            #梯形公式
            weights = torch.ones((x_size, t_size))
            weights[0, :] *= 0.5
            weights[-1, :] *= 0.5
            weights[:, 0] *= 0.5
            weights[:, -1] *= 0.5
            weights = weights.flatten()
            diag_matrix = (self.x[-1]-self.x[0])/(x_size-1) * (self.t[-1]-self.t[0])/(t_size-1) * torch.diag(weights).to(self.device)
            return diag_matrix
        
        weight_matrix=diag_weight(self.x_num_samples, self.t_num_samples)
        #D.T W D
        M_weight_list_experts=[]
        for i in range(len(d_matrix_list)):
            M_weight = torch.matmul(d_matrix_list[i].t(), weight_matrix)
            M_weight = torch.matmul(M_weight, d_matrix_list[i])
            M_weight_list_experts.append(M_weight)
        self.M_weight_list_experts = M_weight_list_experts
        # y_np = d_matrix.cpu().numpy()
        # print("moe ouput:", y_np.shape)
        return  M_weight_list_experts
        
    def experts_rank_mlp(self):
        rank_list_experts=[]
        for i in range(len(self.M_weight_list_experts)):
            eigvals = torch.linalg.eigvalsh(self.M_weight_list_experts[i])

            # 统计大于 epsilon 的特征值数量
            count = (eigvals > self.epsilon).sum().item()
            rank_list_experts.append(count)
        rank_list_experts.append(sum(rank_list_experts[:-1]) - rank_list_experts[-1])
        self.rank_list_experts = rank_list_experts
        return rank_list_experts
    
class PartialMOE(nn.Module):
    def __init__(self, model, n,index=0):
        super().__init__()
        self.model = model
        self.n = n
        self.index=index

    def forward(self, x,training,moe_traing=True):
        if moe_traing:
            x,_=self.model.model(x,training)
            return x
        else:
            for i, layer in enumerate(self.model.model[:self.n+self.index]):
                x = layer(x)
            return x
    

class PartialExpert(nn.Module):
    def __init__(self, model, n):
        super().__init__()
        self.model = model
        self.n = n
        # self.layer1=self.model.model[0]
        self.expert=self.model.model.experts[n]

    def forward(self, x):
        
        x = self.expert(x)

        return x