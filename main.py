# ============================================================================
# Author: klae-zhou
# Date   : 2025-06-05
#  Description:
#   This is a demo for mixture of experts.
# ============

import torch
from torch import nn
import torch.optim as optim
import argparse
import numpy as np
from functorch import make_functional, vmap
from moe import MOE_Model
torch.set_default_dtype(torch.float64)
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:7" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=50)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--opt_steps", type=int, default=1000)
    parser.add_argument("--function", type=str, default="cosx+cos2x+cos3x", help="function")
    parser.add_argument("--interval", type=str, default="[-1,1]")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--loss_coef", type=float, default=1e-3)
    return parser.parse_args()
def get_loss_fn(name):
    if name == "mse":
        return nn.MSELoss()
    elif name == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")
def get_optimizer(name, model_params, lr=1e-3):
    name = name.lower()
    if name == "adam":
        return optim.Adam(model_params,lr=lr,   betas=(0.9, 0.98), eps=1e-9,weight_decay=1e-4)
    elif name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    elif name == "adamw":
        return optim.AdamW(model_params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    


def train_loop(x, y, model,loss_fn, optim, steps=100):
    
    for step in range(steps):
        # 确保在训练模式
        y_hat, aux_loss = model(x)
        # calculate prediction loss
        loss = loss_fn(y_hat, y)
        # combine losses
        total_loss = loss + aux_loss
        optim.zero_grad()
        total_loss.backward()
        optim.step()
        if step % 10 == 0 or step == steps - 1:
            print(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -aux_loss: {aux_loss.item():.8f}")
    return model
            
def eval_model(x, y, model, loss_fn):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x)
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss
    print("Evaluation Results - loss: {:.8f}, aux_loss: {:.8f}".format(loss.item(), aux_loss.item()))

def _init_data_dim1(func: str, interval: str, num_samples: int,device):
    
    interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
    x = (interval[1] - interval[0]) * torch.rand(num_samples, 1) + interval[0]

    # 定义函数解析器
    def parse_function(expr: str):
        expr = expr.replace("cos3x", "np.cos(3*x_np)")
        expr = expr.replace("cos2x", "np.cos(2*x_np)")
        expr = expr.replace("cosx", "np.cos(x_np)")
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



        
def main():
    args=parse_args()
    #init data and model
    data_x,data_y=_init_data_dim1(args.function,args.interval,args.num_samples,args.device)
    print(f"data_x shape: {data_x.shape},data_y shape: {data_y.shape} ")
    model_moe=MOE_Model(args.input_size, args.num_experts,args.hidden_size,args.depth, args.output_size,args.k,args.loss_coef).to(args.device)

    loss_fn =get_loss_fn(args.lossfn)
    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model=train_loop(data_x, data_y, model_moe,loss_fn, optimizer, args.opt_steps)
    eval_model(data_x, data_y, model, loss_fn)
    print("Gates Linear Layer:\n", model.model[0].gating_network.net[0])
    print("Weight Matrix:\n", model.model[0].gating_network.net[0].weight.detach().cpu().numpy())
    print("Bias Vector:\n", model.model[0].gating_network.net[0].bias.detach().cpu().numpy())
    return 


if __name__ == "__main__":
    main()      
    