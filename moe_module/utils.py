import logging
import ast
import time
from functools import wraps
import argparse
import torch
import math
from torch import nn
import torch.optim as optim
import  numpy as np
import matplotlib.pyplot as plt
import sys, os
from torch.quasirandom import SobolEngine
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from data import *
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:5" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
    parser.add_argument("--input_size", type=int, default=2,help="Input size (funcion approximation x) ")
    parser.add_argument("--output_size", type=int, default=1,help="Output size (funcion approximation y=u(x)) ")
    parser.add_argument("--num_experts", type=int, default=2,help="Number of experts")
    parser.add_argument("--hidden_size", type=int, default=20,help="Hidden size of each expert")
    parser.add_argument("--depth", type=int, default=3,help="Depth of the MOE model")
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--opt_steps", type=int, default=40000)
    parser.add_argument("--activation", type=str, default="tanh", help="activation_function")
    parser.add_argument("--init_func", type=str, default="-sin(pi*x)", help="function")
    parser.add_argument("--x_interval", type=str, default="[-1,1]")
    parser.add_argument("--t_interval", type=str, default="[0,1]")
    parser.add_argument("--x_num_samples", type=int, default=100)
    parser.add_argument("--t_num_samples", type=int, default=100)
    parser.add_argument("--k", type=int, default=1,help="top-k selection")
    parser.add_argument("--loss_coef", type=float, default=0)
    parser.add_argument("--x_integral_sample", type=int, default=200, help="x_integral_sample")
    parser.add_argument("--t_integral_sample", type=int, default=100, help="t_integral_sample")
    parser.add_argument("--nu", type=float, default=0.01/np.pi,help="nu in equation u_t + u*u_x - nu*u_xx=0")
    parser.add_argument("--plt_r", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-3,help="epsilon for rank")
    parser.add_argument("--loss_coef_init", type=float, default=20)
    parser.add_argument("--loss_coef_bnd", type=float, default=20)
    parser.add_argument("--vtn",type=int,default=100)
    parser.add_argument("--vxn",type=int,default=100)
    parser.add_argument("--gt",type=torch.Tensor,default=None,help="ground truth")
    parser.add_argument("--X_test",type=torch.Tensor,default=None,help="test data")
    parser.add_argument("--lr_decay",type=float,default=0.98,help="lr decay")
    parser.add_argument("--lbfgs_steps",type=int,default=0,help="number of steps for lbfgs")
    parser.add_argument("--seed",type=int,default=1234)
    return parser.parse_args()
def _init_data_dim1(func: str, x_interval: str, t_interval: str, x_num_samples: int, t_num_samples: int, device,args):
    """
    初始化数据 
    """
    target = x_num_samples * t_num_samples
    x_lo, x_hi = ast.literal_eval(x_interval)
    t_lo, t_hi = ast.literal_eval(t_interval)

    torch.set_default_dtype(torch.float64)
    device = torch.device(device)
    # boundary
    # 等距网格（用于初值与边界）
    x_lin = torch.linspace(x_lo, x_hi, x_num_samples*3, device=device, dtype=torch.get_default_dtype()).view(-1, 1)  # [Nx,1]
    t_lin = torch.linspace(t_lo, t_hi, t_num_samples*3, device=device, dtype=torch.get_default_dtype()).view(-1, 1)  # [Nt,1]

    #  初值面：t = t0（等距 x）
    t0 = t_lin[0, 0]
    X_init = torch.stack([x_lin.squeeze(1), torch.full_like(x_lin.squeeze(1), t0)], dim=1)  # [Nx,2]

    xmin, xmax = x_lin[0, 0], x_lin[-1, 0]
    X_bnd_left  = torch.stack([torch.full_like(t_lin.squeeze(1), xmin), t_lin.squeeze(1)], dim=1)  # [Nt,2]
    X_bnd_right = torch.stack([torch.full_like(t_lin.squeeze(1), xmax), t_lin.squeeze(1)], dim=1)  # [Nt,2]
    X_bnd = torch.cat([X_bnd_left, X_bnd_right], dim=0)  # [2*Nt,2]

    # Interior

    ############################
    #######euqi-steps############
    x_lin = torch.linspace(x_lo, x_hi, (x_num_samples+2)*2, device=device, dtype=torch.get_default_dtype()).view(-1, 1)
    t_lin = torch.linspace(t_lo, t_hi, (t_num_samples+2)*2, device=device, dtype=torch.get_default_dtype()).view(-1, 1)
    

    if x_num_samples >= 3 and t_num_samples >= 2:
        # 内部 x：去掉两端；内部 t：去掉 t0
        x_in = x_lin[1:-1].squeeze(1)   # [Nx-2]
        t_in = t_lin[1:  ].squeeze(1)   # [Nt-1]

        # 生成网格并展平为 [n_f, 2]
        Xg, Tg = torch.meshgrid(x_in, t_in, indexing='ij')   # X 优先
        X_f = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1)  # [(Nx-2)(Nt-1), 2]
    else:
        X_f = torch.zeros((0, 2), device=device, dtype=torch.get_default_dtype())
    
    idx=torch.randint(0,X_f.size(0) , (target,), device=X_f.device)
    X_f=X_f[idx]

    # interface
    # t_interface = torch.linspace(t_lo, t_hi, x_num_samples).view(-1, 1)
    # x_interface = torch.full_like(t_interface,0)
    # X_interface = torch.cat([x_interface, t_interface], dim=1).to(device)
    # # 边界和内点合并
    # X_f = torch.cat([X_f, X_interface], dim=0)
    X_f = torch.unique(X_f, dim=0)
    X_total = torch.cat([X_init, X_bnd, X_f], dim=0)
    X_total = torch.unique(X_total, dim=0)
    
    # Initial
    # 定义函数解析器
    def parse_function(expr: str):
        expr = expr.replace("-sin(pi*x)", "-torch.sin(math.pi*x)")
        def f(x_tensor):
            x=x_tensor
            y_np = eval(expr)
            return y_np
        return f

    f = parse_function(func)
    x= X_init[:, 0:1] 
    u_init = f(x)

    X_init = X_init.to(device)
    X_bnd = X_bnd.to(device)
    X_f = X_f.to(device)
    X_total = X_total.to(device)
    u_init = u_init.to(device)
    return X_init, X_bnd, X_f, X_total, u_init

def get_optimizer(name, model_params, lr=1e-3):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model_params,lr=lr)
    elif name == "sgd":
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    elif name == "adamw":
        return optim.AdamW(model_params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

def get_loss_fn(name):
    if name == "mse":
        return nn.MSELoss()
    elif name == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss function: {name}")
# 日志配置（只需设置一次，通常在 main.py 顶部）
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# 日志装饰器
def log_with_time(func):
    @wraps(func)  # 保留函数原有信息
    def wrapper(*args, **kwargs):
        logger.info(f"【开始】{func.__name__}")
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.exception(f"【异常】{func.__name__} 执行出错：{e}")
            raise  # 继续抛出异常
        finally:
            end = time.time()
            logger.info(f"【完成】{func.__name__}，耗时：{end - start:.4f} 秒")
    return wrapper

def plot_dual_axis(loss: np.ndarray, rank: np.ndarray, step: int,name):
    if rank is None:
        rank=np.zeros(step)
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
    plt.savefig(f"loss_vs_rank_{name}.png")  # 保存图像
    plt.show()

def plot_expert_useless_rank(loss: np.ndarray, rank: np.ndarray, step: int,name):
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
    ax2.plot(z, rank_interp, 'r--', label='Useless Rank')
    ax2.set_ylabel('Useless Rank', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    plt.title("Loss vs. Useless Rank")
    plt.savefig(f"loss_vs_useless_rank_{name}.png")  # 保存图像
    plt.show()

def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "gelu":
        return nn.GELU()
    elif name == "sin":
        return torch.sin
    else:
        raise ValueError(f"Unsupported activation: {name}")



import scipy.io
import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_data(vxn, vtn, nu, writer):
    """
    读取真实 Burgers 方程解 (来自 burgers_shock.mat)
    返回:
      gt: 真解 (torch.Tensor [N,1])
      X_test: 对应网格点 [x,t] (torch.Tensor [N,2])
    """
    # === 1️⃣ 读取真实解 ===
    mat_path = "/home/zhy/Zhou/mixture_of_experts/data/burgers_shock.mat"
    data = scipy.io.loadmat(mat_path)

    # 数据结构说明：
    # data['x'] shape = (Nx,1)
    # data['t'] shape = (1,Nt) or (Nt,1)
    # data['usol'] shape = (Nx,Nt)
    x = data['x'].flatten()
    t = data['t'].flatten()
    u_exact = np.real(data['usol']).T   # 转成 [Nt, Nx] 对应 (t,x) 排列

    # === 2️⃣ 构造 torch 网格 ===
    # 选取等距采样子集（若 vxn,vtn < 原文件分辨率）
    x_idx = np.linspace(0, len(x)-1, vxn, dtype=int)
    t_idx = np.linspace(0, len(t)-1, vtn, dtype=int)
    x_sel = x[x_idx]
    t_sel = t[t_idx]
    u_sel = u_exact[t_idx][:, x_idx]    # shape [vtn, vxn]

    # 构造 torch 张量
    vx_torch = torch.tensor(x_sel, dtype=torch.get_default_dtype()).view(-1, 1)
    vt_torch = torch.tensor(t_sel, dtype=torch.get_default_dtype()).view(-1, 1)
    X, T = torch.meshgrid(vx_torch.squeeze(), vt_torch.squeeze(), indexing='ij')
    X_test = torch.stack([X.flatten(), T.flatten()], dim=1)

    # flatten 真解
    u_flat = u_sel.T.flatten()   # 注意转置确保与 X.flatten() 顺序一致
    gt = torch.tensor(u_flat, dtype=torch.get_default_dtype()).unsqueeze(-1)

    # === 3️⃣ 绘制真解 ===
    fig, ax = plt.subplots(figsize=(17, 9))
    im = ax.contourf(
        T, X, u_sel, levels=100, cmap='coolwarm', vmin=-1, vmax=1
    )
    ax.set_title(f"Ground Truth Burgers Solution (ν={nu:.5f})")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    plt.savefig(f"burgers_gt_nu{nu:.5f}.png", dpi=300)
    print(f"✅ Figure saved as burgers_gt_nu{nu:.5f}.png")

    plt.show()
    writer.add_figure("Ground Truth Burgers Solution", fig)

    return gt, X_test

import datetime
def save_model(model,name='mlp'):
    # 生成时间字符串
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 拼出文件名
    save_path ="/home/zhy/Zhou/mixture_of_experts/saved_model/model_"+name+f"{t}.pth"

    # 保存模型参数
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到: {save_path}")
    
def gates_image(model,X_test,writer):
    """for moe mode MOE_modify_beta"""
    moe=model.moe
    with torch.no_grad():
        gate_output,_,_=moe.gating_network(X_test,train=False)
        gate_output= gate_output/(moe.epsilon+1e-1*torch.abs(gate_output)) #+1e-4*torch.abs(gates)
        gate_output= moe.softmax(gate_output)

    x = X_test[:,0].detach().cpu().numpy()
    t = X_test[:,1].detach().cpu().numpy()
    P = gate_output.detach().cpu().numpy()           # [N, E]
    E = P.shape[1]

    # 还原网格（确保 N == nx*nt）
    ux = np.unique(x); ut = np.unique(t)
    nx, nt = len(ux), len(ut)
    assert nx*nt == len(x), "X_test 不是规则网格（用下面的散点版画法）"

    # 变形为 (nx, nt, E)
    P_grid = P.reshape(nx, nt, E)
    Xg, Tg = np.meshgrid(ux, ut, indexing='ij')

    # 2.1 每个 expert 的热力图
    cols = min(E, 4)
    rows = int(np.ceil(E/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), squeeze=False)
    for i in range(E):
        ax = axes[i//cols][i%cols]
        im = ax.contourf(Xg, Tg, P_grid[:,:,i], levels=100, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Expert {i} prob")
        ax.set_xlabel("x"); ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
    plt.savefig("gates distribution.png")
    writer.add_figure("gates_distribution", fig)

    plt.tight_layout(); plt.show()
        
    return 


    
    

    