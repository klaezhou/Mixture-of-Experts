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
from moe_module.utils import parse_args, _init_data_dim1, get_optimizer, get_loss_fn, log_with_time,plot_dual_axis
from moe_module.epi_rank import epi_rank_moe,epi_rank_mlp
# download data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float64)



@log_with_time
def train_loop(x, y, model,loss_fn, optim, args,steps=100,moe_training=True):
    aux_loss=0
    total_loss_list=[]
    total_rank_list=[]
    for step in range(steps):
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
            if moe_training:
                rank_moe=epi_rank_moe(model,args.interval,args.integral_sample)
                rank=rank_moe.rank_moe()
                total_rank_list.append(rank)
                rank_mlp=epi_rank_mlp(model,args.interval,args.integral_sample)
                rank_list=rank_mlp.rank_mlp()
                print(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -aux_loss: {aux_loss.item():.8f}-rank: {rank},{rank_list}")
            else:
                rank_mlp=epi_rank_mlp(model,args.interval,args.integral_sample,False,1)
                rank=rank_mlp.rank_mlp()
                total_rank_list.append(rank[args.plt_r])
                print(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -rank: {rank}")
    return model,total_loss_list,total_rank_list
            
def eval_model(x, y, model, loss_fn,moe_training=True):
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



def main():
    args=parse_args()
    #init data and model
    data_x,data_y=_init_data_dim1(args.function,args.interval,args.num_samples,args.device)
    print(f"data_x shape: {data_x.shape},data_y shape: {data_y.shape} ")
    model_moe=MOE_Model(args.input_size, args.num_experts,args.hidden_size,args.depth, args.output_size,args.k,args.loss_coef).to(args.device)
    loss_fn =get_loss_fn(args.lossfn)
    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model,total_loss_list_moe,rank_list_moe=train_loop(data_x, data_y, model_moe,loss_fn, optimizer, args,args.opt_steps)
    eval_model(data_x, data_y, model, loss_fn)
    torch.set_printoptions(threshold=float('inf'))
    print("Gates:\n", model.model[0].gates_check)
    model_mlp=MLP_Model(args.input_size, args.hidden_size,args.depth, args.output_size).to(args.device)
    optimizer_mlp=get_optimizer(args.optim,model_mlp.parameters(), lr=args.lr)
        
    model_mlp,total_loss_list_mlp,rank_list_mlp=train_loop(data_x, data_y, model_mlp,loss_fn, optimizer_mlp, args,args.opt_steps,moe_training=False)
    eval_model(data_x, data_y, model_mlp, loss_fn,moe_training=False)
    
    
    plot_dual_axis(np.array(total_loss_list_moe),np.array(rank_list_moe),args.opt_steps,"moe")
    plot_dual_axis(np.array(total_loss_list_mlp),np.array(rank_list_mlp),args.opt_steps,"mlp")


if __name__ == "__main__":
    main()      
    