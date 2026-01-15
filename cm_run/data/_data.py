
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
class data_generator():
    
    def __init__(self,num_samples=1000,device='cpu',interval="[-1,1]"):
        self.num_samples=num_samples
        interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
        # x = (interval[1] - interval[0]) * torch.rand(num_samples, 1) + interval[0]
        x=torch.linspace(interval[0], interval[1], num_samples).view(-1, 1)
        # 定义函数解析器
        def piecewise_hard(x: torch.Tensor):
    
            y = torch.empty_like(x, dtype=torch.float64)

            # 1) 振荡段2
            m1 = (x >= -1.0) & (x < -0.3) # 0-0.1
            # y[m1] = 1/2*torch.cos(1*np.pi*x[m1])+1/2
            y[m1] = torch.cos(3*np.pi*x[m1])
            # y[m1] = -1/torch.abs(x[m1]-1)*torch.cos(10*np.pi*x[m1])
            # 2) 常值段（与左侧跳跃）
            m2 = (x >= -0.3) & (x <= 1)
            # y[m2] = 1/2*torch.cos(1*np.pi*x[m2])
            # y[m2] = 1/torch.abs(x[m2]+1)*torch.cos(10*np.pi*x[m2])
            # m3 = (x >= 0.3) & (x <= 1)
            y[m2] = -torch.cos(3*np.pi*x[m2])
            # y[m3] = 1/np.abs(x[m3]+1)*torch.sin(2*np.pi*x[m3])
            return y
        f=piecewise_hard
        y = f(x)
        x = x.to(device)
        y = y.to(device)
        self.x=x
        self.y=y
        
    def get_data(self):
        return self.x,self.y
    
    
class plot_data():
    def __init__(self,x,y_true,model):
        self.x=x
        self.y_true=y_true
        self.model=model
        self.y=model(x)
        
        
    def plot(self,path):
        x=self.x.cpu().detach().numpy()
        y=self.y.cpu().detach().numpy()
        plt.figure(figsize=(10,6))
        plt.plot(x,self.y_true.cpu().detach().numpy(),label='true function',color='red')
        plt.plot(x,y,label='model prediction',color='blue')
        plt.savefig(f'{path}.png')
        plt.legend()
        plt.show()
        