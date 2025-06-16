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
from moe import MOE_Model
torch.set_default_dtype(torch.float64)
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=20)
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--opt_steps", type=int, default=2000)
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
        return optim.Adam(model_params, lr=lr)
    elif name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    elif name == "adamw":
        return optim.AdamW(model_params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    


def train_loop(x, y, model, loss_fn, optim, steps=100):
    for step in range(steps):
        # 确保在训练模式
        y_hat = model(x, train=True)  # Forward
        loss = loss_fn(y_hat, y)      # Prediction loss
        total_loss = loss             # 可添加辅助损失

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        if step % 10 == 0 or step == steps - 1:
            print(f"Step {step+1}/{steps} - loss: {loss.item():.8f}")
def main():
    args=parse_args()
    model_moe=MOE_Model(input_size=args.input_size, num_experts=args.num_experts, hidden_size=args.hidden_size,output_size=args.output_size)
    x=torch.randn(args.input_size,dtype=torch.float64)
    y=torch.randn(args.hidden_size,dtype=torch.float64)
    loss_fn =get_loss_fn(args.lossfn)
    optimizer = get_optimizer(args.optim, model_moe.parameters(), lr=args.lr)
    train_loop(x, y, model_moe,loss_fn, optimizer, args.opt_steps)
    return 


if __name__ == "__main__":
    main()      
    