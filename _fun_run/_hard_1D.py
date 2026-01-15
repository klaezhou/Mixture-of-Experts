# ============================================================================
# Author: klae-zhou
# Date   : 2025-06-05
#  Description:
#   This is a demo for mixture of experts applied on function 2 approximation.
# ============

import imageio
import torch
from torch import nn
import torch.optim as optim
import argparse
import numpy as np
from functorch import make_functional, vmap
from moe_module.moe import MOE_modify_betap,MLP_Model,Expert
from moe_module.utils import get_optimizer, get_loss_fn, log_with_time, plot_expert_useless_rank
from moe_module.epi_rank import epi_rank_mlp#,epi_rank_expert
# download data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import types

torch.set_default_dtype(torch.float64)



def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=3e-2, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
    parser.add_argument("--input_size", type=int, default=1,help="Input size (funcion approximation x) ")
    parser.add_argument("--output_size", type=int, default=1,help="Output size (funcion approximation y=u(x)) ")
    parser.add_argument("--num_experts", type=int, default=2,help="Number of experts")
    parser.add_argument("--hidden_size", type=int, default=2,help="Hidden size of the MLP")
    parser.add_argument("--depth", type=int, default=1,help="Depth of the MOE model")
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--opt_steps", type=int, default=6000)
    parser.add_argument("--opt_steps_gate", type=int, default=9000)
    parser.add_argument("--function", type=str, default="func2", help="function")
    parser.add_argument("--interval", type=str, default="[-1,1]")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--k", type=int, default=1,help="top-k selection")
    parser.add_argument("--loss_coef", type=float, default=0)
    parser.add_argument("--integral_sample", type=int, default=300, help="integral_sample")
    parser.add_argument("--plt_r", type=int, default=1)
    parser.add_argument("--decrease_rate", type=float, default=0.9)
    parser.add_argument("--index", type=int, default=0,help="index of the expert")
    parser.add_argument("--smooth_steps",type=int,default=13,help="number of steps for smooth mode")
    parser.add_argument("--smooth_lb",type=int,default=80000,help="number lower bound of steps for smooth mode")
    parser.add_argument("--seed",type=int,default=1234) #1234
    parser.add_argument("--fig_index", type=int, default=0)
    parser.add_argument("--ep",type=torch.tensor,default=torch.tensor([[1e-2]],device="cuda:3" if torch.cuda.is_available() else "cpu")) #1234
    parser.add_argument("--ep1",type=torch.tensor,default=torch.tensor([[-1e-2]],device="cuda:3" if torch.cuda.is_available() else "cpu")) #1234
    return parser.parse_args()


    
def piecewise_hard(x: torch.Tensor):
    y = torch.empty_like(x, dtype=torch.float64)

    # 1) 振荡段
    m1 = (x >= -1.0) & (x < 0.1) # 0-0.1
    y[m1] = 1/2*torch.cos(1*np.pi*x[m1])+1/2
    # y[m1] = torch.cos(2*np.pi*x[m1])
    # y[m1] = -1/torch.abs(x[m1]-1)*torch.cos(10*np.pi*x[m1])
    # 2) 常值段（与左侧跳跃）
    m2 = (x >= 0.1) & (x <= 1)
    y[m2] = 1/2*torch.cos(1*np.pi*x[m2])
    # y[m2] = -1/torch.abs(x[m2]+1)*torch.cos(5*np.pi*x[m2])
    # m3 = (x >= 0.3) & (x <= 1)
    # # y[m2] = torch.cos(5*np.pi*x[m2])
    # y[m3] = 1/np.abs(x[m3]+1)*torch.sin(2*np.pi*x[m3])
    return y
def plot_dual_axis(loss: np.ndarray, rank: np.ndarray, step: int,name):
    z = np.arange(1, step + 1)

    # 插值处理 rank
    if rank is None:
        rank=np.zeros(step)
    else:
        from scipy.interpolate import interp1d
        interp = interp1d(np.linspace(0, step - 1, len(rank)), rank, kind='linear', fill_value="extrapolate")
        rank = interp(np.arange(step))

    fig, ax1 = plt.subplots(figsize=(17, 9))

    ax1.plot(z, loss, 'b-', label='Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color='b')
    ax1.set_yscale('log', base=10) 
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()  # 创建共享 x 轴的第二个 y 轴
    ax2.plot(z, rank, 'r--', label='Rank')
    ax2.set_ylabel('Rank', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    plt.title("Loss vs. Rank")
    plt.savefig(f"loss_vs_rank_func2_{name}.png")  # 保存图像
    plt.show()
def _init_data_dim1(func: str, interval: str, num_samples: int,device):
    """
    初始化数据 
    """
    interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
    # x = (interval[1] - interval[0]) * torch.rand(num_samples, 1) + interval[0]
    x=torch.linspace(interval[0], interval[1], num_samples).view(-1, 1)
    # 定义函数解析器
    if func=="func2":
        f=piecewise_hard
    y = f(x)
    x = x.to(device)
    y = y.to(device)
    return x, y
@log_with_time
def train_loop(x, y, model,loss_fn, optim, args,steps=100,moe_training=True):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.996)
    aux_loss=0
    total_loss_list=[]
    total_rank_list=[]
    total_useless_expert_rank=[]
    step_count=args.smooth_steps
    for step in range(steps):
        
        model.train()
        # if moe_training :
        #     step_count -=1
        #     if model.moe.smooth and step_count<=0:
        #         model.moe.smoothing(step,args.smooth_lb)
        #         step_count=args.smooth_steps
        #         # for p in model.moe.experts.parameters():
        #         #     p.requires_grad = False
        #     elif step_count<=0 :
                
        #         model.moe.smoothing(step,args.smooth_lb)
        #         step_count=args.smooth_steps
                
        if moe_training:
            y_hat, aux_loss = model(x)
        else:
            y_hat=model(x)
        # calculate prediction loss
        loss = loss_fn(y_hat, y)
        # combine lossesss
        if moe_training:
            if model.moe.smooth:
                total_loss = loss + aux_loss
            else: total_loss = loss + aux_loss
        else:
            total_loss = loss
        total_loss_list.append(total_loss.item())
        optim.zero_grad() 
        total_loss.backward() 
        optim.step()
        scheduler.step()
        if step % 200 == 0 or step == steps - 1:
            print(f"Step {step+1}/{steps} - loss: {loss.item():.8f}")
            eval_model(args.data_x, args.data_y, model, loss_fn,moe_training,args)
    return model,total_loss_list
            
def eval_model(x, y, model, loss_fn,moe_training=True,args=None):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    if moe_training:
        # model.model.smooth=False
        y_hat, aux_loss = model(x)
    else:
        y_hat=model(x)
    loss = loss_fn(y_hat, y)
    if moe_training:
        total_loss = loss + aux_loss
        y_ep,_ =model(args.ep)
        y_ep1 ,_ =model(args.ep1)
        print("MoE_Model Evaluation Results - loss: {:.12f}, aux_loss: {:.12f}, max error+: {:.10f},error-: {:.12f}".format(loss.item(), aux_loss.item(), torch.abs(y_ep-piecewise_hard(args.ep)).item(),torch.abs(y_ep1-piecewise_hard(args.ep1)).item()))

    else:
        loss=loss
        print("MLP_Model Evaluation Results - loss: {:.12f}, max error: {:.10f}".format(loss.item(),torch.abs(model(args.ep)- piecewise_hard(args.ep)).item()))
        print("error <0: {:.12f}".format(torch.abs(model(args.ep1)- piecewise_hard(args.ep1)).item()))
        
    fig, ax1 = plt.subplots(figsize=(17, 9))
    
    ax1.plot(x.cpu().numpy(), y_hat.detach().cpu().numpy(), label='Prediction')
    ax1.plot(x.cpu().numpy(), y.cpu().numpy(), label='Ground Truth')
    fig.tight_layout()
    plt.title("func_pred")
    if moe_training:
        plt.savefig(f"func_pred_moe.png")  # 保存图像
    else:
        plt.savefig(f"func_pred_mlp.png")
    plt.show()

    if moe_training:
        # Plot experts functions
        fig, ax2 = plt.subplots(figsize=(17, 9))
        ax2.plot(x.cpu().numpy(), y.cpu().numpy(), label='Ground Truth', color='k')
        for i in range(model.model.num_experts):
            experts_outputs = model.model.experts[i](x)
            # plt.subplot(1, model.model[0].num_experts, i + 1)
            ax2.plot(x.cpu().numpy(), experts_outputs.detach().cpu().numpy(), label=f'Expert {i+1}')
            plt.title(f'Experts Function')
        if model.model.smooth :
            gate_output,_,_=model.model.gating_network(x,train=False)
            gates= model.model.soft_topk(gate_output,model.model.k)
        else:
            gates,load= model.model.topkGating(x,False)


        for i in range(model.model.num_experts):
            ax2.plot(x.cpu().numpy(), gates[:,i].detach().cpu().numpy(), label=f'gates {i}')
        plt.legend()

        fig.tight_layout()
        plt.savefig(f"experts_functions2.png")  # 保存图像
        plt.savefig(f"figure/hard_1D_{args.fig_index}.png")
        args.fig_index+=1
        plt.show()

def new_topkGating(self,x,train):
        s,clean,noisy_stddev=self.gating_network(x,train)
        values=s
        values=self.softmax(values)
        values, indices= torch.topk(values,k=self.k,dim=-1) 
        # values= values / (values.sum(1, keepdim=True) + 1e-8) 
        zeros= torch.zeros_like(s)
        gates=zeros.scatter(1, indices, values)
        load=0
        return gates, load
def save_gif(args):


    images = []
    for step in range (args.fig_index):
            images.append(imageio.imread(f"figure/hard_1D_{step}.png"))  
    imageio.mimsave('hard_1D.gif', images, fps=5)

def main():
    
    args=parse_args()
    torch.manual_seed(args.seed)
    #init data and model
    data_x,data_y=_init_data_dim1(args.function,args.interval,args.num_samples,args.device)
    print(f"data_x shape: {data_x.shape},data_y shape: {data_y.shape} ")
    loss_fn =get_loss_fn(args.lossfn)
    args.data_x, args.data_y=data_x, data_y
    train_index = (data_x[:, 0] <= 0) | (data_x[:, 0] > 0)
    data_x_test=data_x[train_index,:]
    data_y_test=data_y[train_index,:]
    
    # #%%

    
    model_moe=MOE_modify_betap(args.input_size, args.num_experts,args.hidden_size,args.depth+1, args.output_size,args.k,args.loss_coef).to(args.device)
    model_moe.moe.smooth=False
    model_moe.moe.topkGating=types.MethodType(new_topkGating, model_moe.moe)


    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model,total_loss_list_moe=train_loop(data_x_test, data_y_test, model_moe,loss_fn, optimizer, args,args.opt_steps_gate)
    eval_model(args.data_x, args.data_y, model, loss_fn,moe_training=True,args=args)
    plot_dual_axis(np.array(total_loss_list_moe),None,args.opt_steps_gate,"moe_fixed")
    # save_gif(args)

    
    #%%

    # model_mlp=MLP_Model(args.input_size, args.hidden_size*2,args.depth+1, args.output_size).to(args.device)
    # optimizer_mlp=get_optimizer(args.optim,model_mlp.parameters(), lr=args.lr)
    # model_mlp,total_loss_list_mlp=train_loop(data_x, data_y, model_mlp,loss_fn, optimizer_mlp, args,args.opt_steps_gate,moe_training=False)
    # eval_model(data_x, data_y, model_mlp, loss_fn,moe_training=False,args=args)
    # plot_dual_axis(np.array(total_loss_list_mlp),None,args.opt_steps_gate   ,"mlp")

    #%%

main()