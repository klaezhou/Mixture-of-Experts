# ============================================================================
# Author: klae-zhou
# Date   : 2025-06-05
#  Description:
#   This is a demo for mixture of experts applied on function 2 approximation.
# ============

import torch
from torch import nn
import torch.optim as optim
import argparse
import numpy as np
from functorch import make_functional, vmap
from moe_module.moe import MOE_Model,MLP_Model
from moe_module.utils import get_optimizer, get_loss_fn, log_with_time
from moe_module.epi_rank import epi_rank_mlp,epi_rank_expert
# download data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
import torch.optim as optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
    parser.add_argument("--input_size", type=int, default=1,help="Input size (funcion approximation x) ")
    parser.add_argument("--output_size", type=int, default=1,help="Output size (funcion approximation y=u(x)) ")
    parser.add_argument("--num_experts", type=int, default=5,help="Number of experts")
    parser.add_argument("--hidden_size", type=int, default=80,help="Hidden size of the MLP")
    parser.add_argument("--depth", type=int, default=3,help="Depth of the MOE model")
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--opt_steps", type=int, default=40000)
    parser.add_argument("--function", type=str, default="cos5x+sin100x+cos30x^2", help="function")
    parser.add_argument("--interval", type=str, default="[-1,1]")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--k", type=int, default=2,help="top-k selection")
    parser.add_argument("--loss_coef", type=float, default=1)
    parser.add_argument("--integral_sample", type=int, default=300, help="integral_sample")
    parser.add_argument("--plt_r", type=int, default=1)
    parser.add_argument("--decrease_rate", type=float, default=0.7)
    parser.add_argument("--index", type=int, default=0,help="index of the expert")
    return parser.parse_args()


def piecewise_hard(x: torch.Tensor):
    # 方波: sign(sin(2πx / period))
    y = torch.sign(torch.sin(10*torch.pi*x))
    return y

def plot_dual_axis(loss: np.ndarray, rank: np.ndarray, step: int,name):
    z = np.arange(1, step + 1)

    # 插值处理 rank
    if len(rank) != step:
        from scipy.interpolate import interp1d
        interp = interp1d(np.linspace(0, step - 1, len(rank)), rank, kind='linear', fill_value="extrapolate")
        rank_interp = interp(np.arange(step))
    else:
        rank_interp = rank

    fig, ax1 = plt.subplots(figsize=(17, 9))

    ax1.plot(z, loss, 'b-', label='Loss')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color='b')
    ax1.set_yscale('log', base=10) 
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()  # 创建共享 x 轴的第二个 y 轴
    ax2.plot(z, rank_interp, 'r--', label='Rank')
    ax2.set_ylabel('Rank', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    plt.title("Loss vs. Rank")
    plt.savefig(f"loss_vs_rank_func2_{name}.png")  # 保存图像
    plt.show()
# def _init_data_dim1(func: str, interval: str, num_samples: int,device):
#     """
#     初始化数据 
#     """
#     interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
#     # x = (interval[1] - interval[0]) * torch.rand(num_samples, 1) + interval[0]
#     x=torch.linspace(interval[0], interval[1], num_samples).view(-1, 1)
#     # 定义函数解析器
    
#     f=piecewise_hard
#     y = f(x)
#     noise_level = 0.05  # 控制噪声幅度，可调
#     # y = y + noise_level * torch.randn_like(y)
    
#     x = x.to(device)
#     y = y.to(device)
#     return x, y

def _init_data_dim1(func: str, interval: str, num_samples: int,device):
    """
    初始化数据 
    """
    interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
    # x = (interval[1] - interval[0]) * torch.rand(num_samples, 1) + interval[0]
    x=torch.linspace(interval[0], interval[1], num_samples).view(-1, 1)
    # 定义函数解析器
    def parse_function(expr: str):
        expr = expr.replace("cos30x^2", "np.cos(30*x_np*x_np)")
        expr = expr.replace("sin100x", "np.sin(100*x_np)")
        expr = expr.replace("cos5x", "np.cos(5*x_np)")
        def f(x_tensor):
            x_np = x_tensor.numpy()
            y_np = eval(expr)
            return torch.from_numpy(y_np)
        return f

    f = parse_function(func)
    y = f(x)
    x = x.to(device)
    y = y.to(device)
    return x, y
@log_with_time
def train_loop(x, y, model,loss_fn, optim, args,steps=100,moe_training=True):
    aux_loss=0
    total_loss_list=[]
    total_rank_list=[]
    total_useless_rank_list=[]
    scheduler = optimizer.lr_scheduler.StepLR(optim, step_size=1000, gamma=0.85)
    for step in range(steps):
        # 确保在训练模式
        if step >=steps:
            model.eval()
        else:
            model.train()
        
        if moe_training:
            y_hat, aux_loss = model(x)
            if step % 1000 == 0 :
                # model.adlosscoff(args.decrease_rate)
                eval_model(x, y, model, loss_fn,moe_training)
        else:
            y_hat=model(x)
            if step % 1000 == 0 :
                eval_model(x, y, model, loss_fn,moe_training)
        # calculate prediction loss
        loss = loss_fn(y_hat, y)
        # combine losses
        if moe_training:
            total_loss = loss + aux_loss
        else:
            total_loss = loss
        total_loss_list.append(total_loss.item())
        optim.zero_grad()
        total_loss.backward()
        optim.step()
           
        scheduler.step()
        
        
        if step % 100 == 0 or step == steps - 1:
            if moe_training:
                rank_expert=epi_rank_expert(model,args.interval,args.integral_sample,args.index)
                rank_expert_list=rank_expert.rank_mlp()
                expert_total=rank_expert.rank_mlp(True)
                useless_rank=sum(rank_expert_list)-sum(expert_total)
                # useless_rank=useless_rank/sum(expert_total)
                total_useless_rank_list.append(useless_rank)
                rank_mlp=epi_rank_mlp(model,args.interval,args.integral_sample)
                rank_list=rank_mlp.rank_mlp()
                total_rank_list.append(rank_list[args.index])
                print(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -aux_loss: {aux_loss.item():.8f} -rank: {rank_list} --expert: {rank_expert_list} --expert total: {expert_total} --useless rank: {useless_rank}")
            else:
                rank_mlp=epi_rank_mlp(model,args.interval,args.integral_sample,False,1)
                rank=rank_mlp.rank_mlp()
                total_rank_list.append(rank[args.plt_r])
                print(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -rank: {rank}")
    return model,total_loss_list,total_rank_list, total_useless_rank_list
            
def eval_model(x, y, model, loss_fn,moe_training=True,args=None):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    if moe_training:
        y_hat, aux_loss = model(x)
    else:
        y_hat=model(x)
    loss = loss_fn(y_hat, y)
    if moe_training:
        total_loss = loss + aux_loss
        print("MoE_Model Evaluation Results - loss: {:.8f}, aux_loss: {:.8f}".format(loss.item(), aux_loss.item()))
    else:
        loss=loss
        print("MLP_Model Evaluation Results - loss: {:.8f}".format(loss.item()))
    
    fig, ax1 = plt.subplots(figsize=(17, 9))
    
    ax1.plot(x.cpu().numpy(), y_hat.detach().cpu().numpy(), label='Prediction')
    ax1.plot(x.cpu().numpy(), y.cpu().numpy(), label='Ground Truth')
    fig.tight_layout()
    plt.title("func_pred")
    if moe_training:
        plt.savefig(f"func2_pred_moe.png")  # 保存图像
    else:
        plt.savefig(f"func2_pred_mlp.png")
    plt.show()

def main():
    args=parse_args()
    #init data and model
    data_x,data_y=_init_data_dim1(args.function,args.interval,args.num_samples,args.device)
    print(f"data_x shape: {data_x.shape},data_y shape: {data_y.shape} ")
    model_moe=MOE_Model(args.input_size, args.num_experts,args.hidden_size,args.depth, args.output_size,args.k,args.loss_coef).to(args.device)
    loss_fn =get_loss_fn(args.lossfn)
    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model,total_loss_list_moe,rank_list_moe,useless_rank_list_experts=train_loop(data_x, data_y, model_moe,loss_fn, optimizer, args,args.opt_steps)
    eval_model(data_x, data_y, model, loss_fn)
    torch.set_printoptions(threshold=float('inf'))
    # print("Gates:\n", model.model[0].gates_check)
    model_mlp=MLP_Model(args.input_size, args.hidden_size,args.depth, args.output_size).to(args.device)
    optimizer_mlp=get_optimizer(args.optim,model_mlp.parameters(), lr=args.lr)
        
    model_mlp,total_loss_list_mlp,rank_list_mlp,_=train_loop(data_x, data_y, model_mlp,loss_fn, optimizer_mlp, args,args.opt_steps*2,moe_training=False)
    eval_model(data_x, data_y, model_mlp, loss_fn,moe_training=False)
    


    plot_dual_axis(np.array(total_loss_list_moe),np.array(rank_list_moe),args.opt_steps,"moe")
    plot_dual_axis(np.array(total_loss_list_mlp),np.array(rank_list_mlp),args.opt_steps*2,"mlp")
    plot_dual_axis(np.array(total_loss_list_moe),np.array(useless_rank_list_experts),args.opt_steps,"useless_experts")

torch.manual_seed(42)
main()      
    