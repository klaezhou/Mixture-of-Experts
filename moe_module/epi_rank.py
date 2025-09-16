import torch
import torch.nn as nn
import numpy as np
#numerical integral to compute epi_rank

class epi_rank_mlp():
    def __init__(self, model,interval,num_samples,moe_training=True,index=1):
        """index: defalut 2 ; if no moe, then 1"""
        self.moe_training=moe_training
        self.model = model
        self.index=index
        #find the MLP in MOE_Model moe_model => moe-*fcnn-ouputlayer, index=2 表示从第一个fcnn开始计算 
        #self.model.model[:self.n+self.index] 表示到前n+index个fcnn
        
        self.mlp = [ PartialMOE(model,n,self.index) for n in range(len(model.model)-1)]
        self.device=next(model.parameters()).device
        self.epsilon =1e-3
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
                    # print("length",len(self.model.model))
                    # print(f"mlp {i}ouput:", d_matrix.shape)
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
            epsilon = self.epsilon

            # 统计大于 epsilon 的特征值数量
            count = (eigvals > epsilon).sum().item()
            rank_list.append(count)
        self.rank_list = rank_list
        return rank_list
            
class epi_rank_expert():
    def __init__(self, model,interval,num_samples,index=0):
        """
        This is a class for computing the rank of an expert in a mixture of experts model. --1D
        Parameters:
        model: class MOE_Model in moe.py
        interval: the interval of the input"
        num_samples: the number of samples
        index: the index of the moe layer, defalut 0
        
        """
        self.model=model
        self.index=index
        interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
        self.device=next(model.parameters()).device
        self.x = torch.linspace(interval[0], interval[1], num_samples).view(-1, 1).to(self.device)
        self.num_experts=self.model.num_experts
        self.num_samples = num_samples
        self.experts=[Expert_output(model,i,self.index) for i in range(self.num_experts)]
        self.M_weight_list=self.compute_matrix_list()
        self.M_weight_total=self.compute_total_matrix()
        self.rank_list=0
        self.total_rank=0
    
    def compute_matrix_list(self):
                d_matrix_list=[]
                for i in range(len(self.experts)):
                    d_matrix=self.experts[i](self.x)
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
                # y_np = d_matrix.cpu().numpy()
                # print("moe ouput:", y_np.shape)
                return  M_weight_list
    
    def compute_total_matrix(self):
        d_matrix_list=[]
        for i in range(len(self.experts)):
                    d_matrix=self.experts[i](self.x)
                    d_matrix=d_matrix.detach()
                    d_matrix_list.append(d_matrix)
        D_matrix=torch.cat(d_matrix_list, dim=1)
        #init weight matrix, numerical
        def diag_weight(size):
            #梯形公式
            weights = torch.ones(size)
            weights[0] = weights[-1] = 0.5
            diag_matrix = torch.diag(weights).to(self.device)
            return diag_matrix
        weight_matrix=diag_weight(self.num_samples)
        #D.T W D
        M_weight=0
        M_weight = torch.matmul(D_matrix.t(), weight_matrix)
        M_weight = torch.matmul(M_weight, D_matrix)
        # y_np = d_matrix.cpu().numpy()
        # print("moe ouput:", y_np.shape)
        return  M_weight
        
        
    def rank_mlp(self,total_gram=False):
        "如果要计算total rank 那么输入total_gram改为true"
        if total_gram:
            gram=self.M_weight_total
        else:
            gram=self.M_weight_list
        
        if isinstance(gram, torch.Tensor):
            gram = [gram]
        rank_list=[]
        for i in range(len(gram)):
            eigvals = torch.linalg.eigvalsh(gram[i])

            # 设定阈值 epsilon
            epsilon = 1e-3

            # 统计大于 epsilon 的特征值数量
            count = (eigvals > epsilon).sum().item()
            rank_list.append(count)
        self.rank_list = rank_list
        return rank_list
    
    
class PartialMOE(nn.Module):
    def __init__(self, model, n,index=1):
        super().__init__()
        self.model = model
        self.n = n # partial model length
        self.index=index #default 1   为了第一个保证【：1】 当n=0时能取到第一层的输出

    def forward(self, x,training,moe_training=True):
        for i, layer in enumerate(self.model.model[:self.n+self.index]):
            if i == 0 and moe_training:
                x, _ = layer(x,training)
            else:
                x = layer(x)
        return x
    
    
class Expert_output(nn.Module):
    def __init__(self, model,n,index):
        super().__init__()
        self.model=model
        self.n=n # expert index
        self.index=index # moe layer index moe层的位置，默认model里只包含一个moe层，其余均为fcnn
        self.mlp=self.model.model[:self.index]
        self.expert=self.model.model[self.index].experts[self.n]
        # print("expert output:",self.expert)
    
    def forward(self,x):
        if self.index!=0:
            x=self.mlp(x)
        x=self.expert(x)
        return x
        
    