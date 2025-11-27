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
from moe_module.utils import get_optimizer, get_loss_fn, log_with_time, plot_expert_useless_rank
from moe_module.epi_rank import epi_rank_mlp#,epi_rank_expert
# download data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)



def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
    parser.add_argument("--input_size", type=int, default=1,help="Input size (funcion approximation x) ")
    parser.add_argument("--output_size", type=int, default=1,help="Output size (funcion approximation y=u(x)) ")
    parser.add_argument("--num_experts", type=int, default=2,help="Number of experts")
    parser.add_argument("--hidden_size", type=int, default=8,help="Hidden size of the MLP")
    parser.add_argument("--depth", type=int, default=1,help="Depth of the MOE model")
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--opt_steps", type=int, default=10000)
    parser.add_argument("--function", type=str, default="func2", help="function")
    parser.add_argument("--interval", type=str, default="[-1,1]")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--k", type=int, default=1,help="top-k selection")
    parser.add_argument("--loss_coef", type=float, default=0.1)
    parser.add_argument("--integral_sample", type=int, default=300, help="integral_sample")
    parser.add_argument("--plt_r", type=int, default=1)
    parser.add_argument("--decrease_rate", type=float, default=0.9)
    parser.add_argument("--index", type=int, default=0,help="index of the expert")
    parser.add_argument("--smooth_steps",type=int,default=20,help="number of steps for smooth mode")
    parser.add_argument("--smooth_lb",type=int,default=4000,help="number lower bound of steps for smooth mode")
    parser.add_argument("--seed",type=int,default=123) #1234
    parser.add_argument("--ep",type=torch.tensor,default=torch.tensor([[5e-3]],device="cuda:3" if torch.cuda.is_available() else "cpu")) #1234
    parser.add_argument("--ep1",type=torch.tensor,default=torch.tensor([[-5e-3]],device="cuda:3" if torch.cuda.is_available() else "cpu")) #1234
    return parser.parse_args()


def piecewise_hard(x: torch.Tensor):
    
    y = torch.empty_like(x, dtype=torch.float64)

    # 1) 振荡段
    m1 = (x >= -1.0) & (x < 0)
    y[m1] = -torch.sin(5*np.pi*x[m1])

    # 2) 常值段（与左侧跳跃）
    m2 = (x >= 0) & (x <= 1)
    y[m2] = torch.cos(5*np.pi*x[m2])

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
        if moe_training :
            step_count -=1
            if model.moe.smooth and step_count<=0:
                model.moe.smoothing(step,args.smooth_lb)
                step_count=args.smooth_steps
            elif step_count<=0 :
                model.moe.smoothing(step,args.smooth_lb)
                step_count=args.smooth_steps
                
        if moe_training:
            y_hat, aux_loss = model(x)
        else:
            y_hat=model(x)
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
        if step % 500 == 0 or step == steps - 1:
            print(f"Step {step+1}/{steps} - loss: {loss.item():.8f}")
            eval_model(x, y, model, loss_fn,moe_training,args)
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
        print("MoE_Model Evaluation Results - loss: {:.12f}, aux_loss: {:.12f}, max error: {:.10f}".format(loss.item(), aux_loss.item(), torch.abs(y_ep- torch.cos(5*np.pi*args.ep)).item()))
        print("error <0: {:.12f}".format(torch.abs(y_ep1+ torch.sin(5*np.pi*args.ep1)).item()))
    else:
        loss=loss
        print("MLP_Model Evaluation Results - loss: {:.12f}, max error: {:.10f}".format(loss.item(),torch.abs(model(args.ep)- torch.cos(5*np.pi*args.ep)).item()))
        print("error <0: {:.12f}".format(torch.abs(model(args.ep1)+ torch.sin(5*np.pi*args.ep1)).item()))
        
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
        for i in range(model.model[0].num_experts):
            experts_outputs = model.model[0].experts[i](x)
            # plt.subplot(1, model.model[0].num_experts, i + 1)
            ax2.plot(x.cpu().numpy(), experts_outputs.detach().cpu().numpy(), label=f'Expert {i+1}')
            plt.title(f'Experts Function')
        if model.model[0].smooth and not model.model[0].fix_gates:
            gates,_= model.model[0].soft_topk_Gating(x,train=True)
            # gates= gates/(self.epsilon) # +1e-1*torch.abs(gates)
            # gates= self.softmax(gates)
            # gates= gates / (gates.sum(1, keepdim=True) + 1e-8) 
        elif not model.model[0].smooth and not model.model[0].fix_gates:
            gates,load= model.model[0].topkGating(x,False)
        elif model.model[0].fix_gates:
            gates,load= model.model[0].fixed_gating(x,False)
        ax2.plot(x.cpu().numpy(), gates[:,0].detach().cpu().numpy(), label=f'gates 0')
        ax2.plot(x.cpu().numpy(), gates[:,1].detach().cpu().numpy(), label=f'gates 1')
        plt.legend()

        fig.tight_layout()
        plt.savefig(f"experts_functions2.png")  # 保存图像
        plt.show()



def main():
    args=parse_args()
    torch.manual_seed(args.seed)
    #init data and model
    data_x,data_y=_init_data_dim1(args.function,args.interval,args.num_samples,args.device)
    print(f"data_x shape: {data_x.shape},data_y shape: {data_y.shape} ")
    loss_fn =get_loss_fn(args.lossfn)
    
    #%%

    model_moe=MOE_Model(args.input_size, args.num_experts,args.hidden_size,args.depth+1, args.output_size,args.k,args.loss_coef).to(args.device)
    
    model_moe.model[0].fix_gates=True
    for p in model_moe.model[0].gating_network.parameters():
        p.requires_grad = False

    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model,total_loss_list_moe=train_loop(data_x, data_y, model_moe,loss_fn, optimizer, args,args.opt_steps)
    eval_model(data_x, data_y, model, loss_fn,moe_training=True,args=args)
    plot_dual_axis(np.array(total_loss_list_moe),None,args.opt_steps,"moe_fixed")

    model_moe.model[0].fix_gates=False
    for p in model_moe.model[0].experts.parameters():
        p.requires_grad = False
    for p in model_moe.model[0].gating_network.parameters():
        p.requires_grad = True

    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model,total_loss_list_moe=train_loop(data_x, data_y, model_moe,loss_fn, optimizer, args,args.opt_steps)
    eval_model(data_x, data_y, model, loss_fn,moe_training=True,args=args)
    plot_dual_axis(np.array(total_loss_list_moe),None,args.opt_steps,"moe")
    
    #%%

    # model_mlp=MLP_Model(args.input_size, args.hidden_size*2,args.depth+1, args.output_size).to(args.device)
    # optimizer_mlp=get_optimizer(args.optim,model_mlp.parameters(), lr=args.lr)
    # model_mlp,total_loss_list_mlp=train_loop(data_x, data_y, model_mlp,loss_fn, optimizer_mlp, args,args.opt_steps,moe_training=False)
    # eval_model(data_x, data_y, model_mlp, loss_fn,moe_training=False,args=args)
    # plot_dual_axis(np.array(total_loss_list_mlp),None,args.opt_steps,"mlp")

    #%%

if __name__ == "__main__":
    
    main()