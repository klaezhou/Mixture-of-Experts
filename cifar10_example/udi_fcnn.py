import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import argparse
from tqdm import tqdm
from model_list import get_network
import math
from moe_module.utils import get_activation, get_loss_fn, get_optimizer, log_with_time,plot_dual_axis
##dataset: https://www.cs.toronto.edu/~kriz/cifar.html.

# 参数解析
parser = argparse.ArgumentParser(description="Train a udi_res_fcnn")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
parser.add_argument("--input_size", type=int, default=10,help="dimension of image")
parser.add_argument("--output_size", type=int, default=10,help="Output classes")
parser.add_argument("--num_experts", type=int, default=10,help="Number of experts")
parser.add_argument("--hidden_size", type=int, default=128,help="Hidden size of the MLP")
parser.add_argument("--depth", type=int, default=10,help="Depth of the MOE model")
parser.add_argument("--lossfn", type=str, default="ce", help="Loss function.")
parser.add_argument("--optim", type=str, default="adamw")
parser.add_argument("--k", type=int, default=4,help="top-k selection")
parser.add_argument("--loss_coef", type=float, default=1e-2)
parser.add_argument("--activation", type=str, default="relu")
parser.add_argument("--model", type=str, default="udi_res_fcnn")
parser.add_argument("--batch_size", type=int, default=128,help="Batch size")
parser.add_argument("--download", type=bool, default=False,help="Download dataset")
args=parser.parse_args()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def random_projection_matrix(k, d, device=args.device):
    # 生成标准正态分布矩阵
    R = torch.randn(k, d, device=device)  # 每个 r_ij ~ N(0, 1)
    R = R / torch.sqrt(torch.tensor(k, dtype=torch.float32, device=device))  # 除以 sqrt(k)
    return R
@log_with_time
def train_loop(train_loader, model,loss_fn, optim,args,R,device=args.device):
    model.train()
    total_loss_list=[]
    for epoch in range(args.epochs):
        running_loss = 0.0 
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)        # [B, 3072]
            labels = labels.to(device)
            # 应用随机投影
            projected = torch.matmul(images, R.T)  # [B, k]
            y_hat=model(projected)
            loss=loss_fn(y_hat, labels)
            total_loss =loss
            total_loss_list.append(total_loss.item())
            optim.zero_grad()
            total_loss.backward()
            optim.step()
            running_loss+=loss.item()
        print(f"[Epoch {epoch+1}/{args.epochs}]-loss: {running_loss/math.ceil(50000/args.batch_size) :.8f}")
    plot_dual_axis(np.array(total_loss_list),np.array([0]),len(total_loss_list),args.model)
    return model,total_loss_list

def eval_model(test_loader, model,R ,device=args.device,classes=classes):
    model.eval()
    correct = 0
    total = 0

    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            projected = torch.matmul(images, R.T)

            outputs= model(projected)  # 只关注分类输出
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                class_correct[label] += (pred == label).item()
                class_total[label] += 1

    acc = 100 * correct / total
    print(f"\n Overall Accuracy: {acc:.4f}%")

    print("\n Per-class Accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"{classes[i]:>6}: {class_acc:.2f}%")
        else:
            print(f"{classes[i]:>6}: No samples.")
    
    
        

device=args.device

# 图像预处理：转换为Tensor，并归一化
transform =transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 最后做
])
train_dataset = datasets.CIFAR10(root='/home/zhy/Zhou/mixture-of-experts/dataset', train=True, 
                                 download=args.download, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_dataset = datasets.CIFAR10(root='/home/zhy/Zhou/mixture-of-experts/dataset', train=False, 
                                download=args.download, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
# 假设你想降维到 k=10 维
k = args.input_size
d = 3 * 32 * 32  # 3072
R = random_projection_matrix(k, d, device=device)
loss_fn=get_loss_fn(args.lossfn)
model=get_network(args).to(device)
optimizer=get_optimizer(args.optim, model.parameters(), lr=args.lr)
model_trained,total_loss_list= train_loop( train_loader, model,loss_fn, optimizer,args,R)
eval_model(test_loader, model_trained,R)

