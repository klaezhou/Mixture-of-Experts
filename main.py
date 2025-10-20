# ============================================================================
# Author: klae-zhou
# Date   : 2025-06-05
#  Description:
#   This is a demo for mixture of experts applied on function approximation.
# ============

import torch
from torch import nn
import torch.optim as optim
import argparse
import numpy as np
from functorch import make_functional, vmap
from moe_module.moe import MOE_Model,MLP_Model
from moe_module.utils import parse_args, _init_data_dim1, get_optimizer, get_loss_fn, log_with_time,plot_dual_axis, plot_expert_useless_rank
from moe_module.epi_rank import epi_rank_moe,epi_rank_mlp
# download data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)



@log_with_time
def train_loop(x, y, model,loss_fn, optim, args,steps=100,moe_training=True):
    aux_loss=0
    total_loss_list=[]
    total_rank_list=[]
    total_useless_expert_rank=[]
    for step in range(steps):
        model.train()
        # 确保在训练模式
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
        if step % 100 == 0 or step == steps - 1:
            eval_model(x, y, model, loss_fn,moe_training,args)
            if moe_training:
                # rank_moe=epi_rank_moe(model,args.interval,args.integral_sample)
                # rank=rank_moe.rank_moe()
                
                rank_mlp=epi_rank_mlp(model,args.interval,args.integral_sample)
                rank_list=rank_mlp.rank_mlp()
                total_rank_list.append(rank_list[0])
                rank_list_experts=rank_mlp.experts_rank_mlp()
                print(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -aux_loss: {aux_loss.item():.8f} -rank: {rank_list[0]},{rank_list} -experts_rank: {rank_list_experts[:-2]} -total_experts_rank: {rank_list_experts[-2]} -useless_expert_rank: {rank_list_experts[-1]}")
                total_useless_expert_rank.append(rank_list_experts[-1])
            else:
                rank_mlp=epi_rank_mlp(model,args.interval,args.integral_sample,False,0)
                rank=rank_mlp.rank_mlp()
                total_rank_list.append(rank[args.plt_r])
                print(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -rank: {rank}")
    return model,total_loss_list,total_rank_list,total_useless_expert_rank
            
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
        plt.savefig(f"func_pred_moe.png")  # 保存图像
    else:
        plt.savefig(f"func_pred_mlp.png")
    plt.show()

# %%
    if moe_training:
        # Plot experts functions
        fig, ax2 = plt.subplots(figsize=(17, 9))
        ax2.plot(x.cpu().numpy(), y.cpu().numpy(), label='Ground Truth', color='k')
        for i in range(model.model[0].num_experts):
            experts_outputs = model.model[0].experts[i](x)
            # plt.subplot(1, model.model[0].num_experts, i + 1)
            ax2.plot(x.cpu().numpy(), experts_outputs.detach().cpu().numpy(), label=f'Expert {i+1}')
            plt.title(f'Experts Function')

        fig.tight_layout()
        plt.savefig(f"experts_functions.png")  # 保存图像
        plt.show()
# %%


def main():
    args=parse_args()
    #init data and model
    data_x,data_y=_init_data_dim1(args.function,args.interval,args.num_samples,args.device)
    print(f"data_x shape: {data_x.shape},data_y shape: {data_y.shape} ")
    model_moe=MOE_Model(args.input_size, args.num_experts,args.hidden_size,args.depth, args.output_size,args.k,args.loss_coef).to(args.device)
    loss_fn =get_loss_fn(args.lossfn)
    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model,total_loss_list_moe,rank_list_moe,total_useless_expert_rank_moe=train_loop(data_x, data_y, model_moe,loss_fn, optimizer, args,args.opt_steps)
    eval_model(data_x, data_y, model, loss_fn)
    torch.set_printoptions(threshold=float('inf'))
    # print("Gates:\n", model.model[1].gates_check)
    model_mlp=MLP_Model(args.input_size, args.hidden_size*args.num_experts,args.depth, args.output_size).to(args.device)
    optimizer_mlp=get_optimizer(args.optim,model_mlp.parameters(), lr=args.lr)
        
    model_mlp,total_loss_list_mlp,rank_list_mlp,total_useless_expert_rank_mlp=train_loop(data_x, data_y, model_mlp,loss_fn, optimizer_mlp, args,args.opt_steps,moe_training=False)
    eval_model(data_x, data_y, model_mlp, loss_fn,moe_training=False)
    
    
    plot_dual_axis(np.array(total_loss_list_moe),np.array(rank_list_moe),args.opt_steps,"moe")
    plot_dual_axis(np.array(total_loss_list_mlp),np.array(rank_list_mlp),args.opt_steps,"mlp")
    plot_expert_useless_rank(np.array(total_loss_list_moe),np.array(total_useless_expert_rank_moe),args.opt_steps,"moe")


if __name__ == "__main__":
    main()      
    