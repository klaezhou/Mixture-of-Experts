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
from torch.quasirandom import SobolEngine
from data import *
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
    parser.add_argument("--lr_decay", type=float, default=0.9987, help="Learning rate decay factor.")
    parser.add_argument("--lr_step", type=int, default=10, help="Learning rate decay step size.")
    parser.add_argument("--device", type=str, default="cuda:5" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
    parser.add_argument("--input_size", type=int, default=2,help="Input size (funcion approximation x) ")
    parser.add_argument("--output_size", type=int, default=1,help="Output size (funcion approximation y=u(x)) ")
    parser.add_argument("--num_experts", type=int, default=5,help="Number of experts")
    parser.add_argument("--hidden_size", type=int, default=10,help="Hidden size of each expert")
    parser.add_argument("--depth", type=int, default=3,help="Depth of the MOE model")
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--opt_steps", type=int, default=20000)
    parser.add_argument("--activation", type=str, default="tanh", help="activation_function")
    parser.add_argument("--init_func", type=str, default="-sin(pi*x)", help="function")
    parser.add_argument("--x_interval", type=str, default="[-1,1]")
    parser.add_argument("--t_interval", type=str, default="[0,1]")
    parser.add_argument("--x_num_samples", type=int, default=400)
    parser.add_argument("--t_num_samples", type=int, default=200)
    parser.add_argument("--k", type=int, default=2,help="top-k selection")
    parser.add_argument("--loss_coef", type=float, default=1)
    parser.add_argument("--x_integral_sample", type=int, default=200, help="x_integral_sample")
    parser.add_argument("--t_integral_sample", type=int, default=100, help="t_integral_sample")
    parser.add_argument("--nu", type=float, default=0.001,help="nu in equation u_t + u*u_x - nu*u_xx=0")
    parser.add_argument("--plt_r", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--loss_coef_init", type=float, default=1)
    parser.add_argument("--loss_coef_bnd", type=float, default=1)
    parser.add_argument("--vtn",type=int,default=150)
    parser.add_argument("--vxn",type=int,default=300)
    parser.add_argument("--gt",type=torch.Tensor,default=None,help="ground truth")
    parser.add_argument("--X_test",type=torch.Tensor,default=None,help="test data")
    return parser.parse_args()
def _init_data_dim1(func: str, x_interval: str, t_interval: str, x_num_samples: int, t_num_samples: int, device):
    """
    初始化数据 
    """
    x_lo, x_hi = ast.literal_eval(x_interval)
    t_lo, t_hi = ast.literal_eval(t_interval)

    dtype = torch.float64
    device = torch.device(device)
    # boundary
    # 等距网格（用于初值与边界）
    x_lin = torch.linspace(x_lo, x_hi, x_num_samples, device=device, dtype=dtype).view(-1, 1)  # [Nx,1]
    t_lin = torch.linspace(t_lo, t_hi, t_num_samples, device=device, dtype=dtype).view(-1, 1)  # [Nt,1]

    #  初值面：t = t0（等距 x）
    t0 = t_lin[0, 0]
    X_init = torch.stack([x_lin.squeeze(1), torch.full_like(x_lin.squeeze(1), t0)], dim=1)  # [Nx,2]

    xmin, xmax = x_lin[0, 0], x_lin[-1, 0]
    X_bnd_left  = torch.stack([torch.full_like(t_lin.squeeze(1), xmin), t_lin.squeeze(1)], dim=1)  # [Nt,2]
    X_bnd_right = torch.stack([torch.full_like(t_lin.squeeze(1), xmax), t_lin.squeeze(1)], dim=1)  # [Nt,2]
    X_bnd = torch.cat([X_bnd_left, X_bnd_right], dim=0)  # [2*Nt,2]

    # Interior
    n_f = max((x_num_samples - 2) * (t_num_samples - 1), 0)
    if n_f > 0:
        sobol = SobolEngine(dimension=2, scramble=True)
        s = sobol.draw(n_f).to(device=device, dtype=dtype)  # in [0,1]^2
        # 映射到 (x_lo, x_hi) × (t_lo, t_hi]，并避免恰落边界
        eps_x = 1e-7
        eps_t = 1e-7
        x_f = x_lo + (x_hi - x_lo) * (s[:, 0:1] * (1 - 2*eps_x) + eps_x)      # (x_lo, x_hi)
        t_f = t_lo + (t_hi - t_lo) * (s[:, 1:2] * (1 - eps_t) + eps_t)        # (t_lo, t_hi]
        X_f = torch.cat([x_f, t_f], dim=1)  # [n_f,2]
    else:
        X_f = torch.zeros((0, 2), device=device, dtype=dtype)

    
    t_interface = torch.linspace(t_lo, t_hi, x_num_samples*5).view(-1, 1)
    x_interface = torch.full_like(t_interface,0)
    X_interface = torch.cat([x_interface, t_interface], dim=1).to(device)
    # 边界和内点合并
    X_f = torch.cat([X_f, X_interface], dim=0)
    X_f = torch.unique(X_f, dim=0)

    # 6) 合并点集并去重
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
        return optim.Adam(model_params,lr=lr,   betas=(0.9, 0.98), eps=1e-9,weight_decay=1e-4)
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

def plot_dual_axis(loss: np.ndarray, rank: np.ndarray, step: int,name, writer):
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
    else:
        raise ValueError(f"Unsupported activation: {name}")
    
def generate_data(vxn,vtn,nu, writer):
    # data points are generated by equispaced grid
    vx=np.linspace(-1.0,1.0,vxn)
    vt=np.linspace(0.0,1.0,vtn) 
    vx_torch=torch.from_numpy(vx).view(-1,1)
    vt_torch=torch.from_numpy(vt).view(-1,1)
    X, T = torch.meshgrid(vx_torch.squeeze(), vt_torch.squeeze(), indexing='ij')
    X_test = torch.stack([X.flatten(), T.flatten()], dim=1)
    
    vu=bg_gt.burgers_viscous_time_exact1 ( nu, vxn, vx, vtn, vt )
    u_flat = vu.flatten()  # 先转置再flatten
    gt= torch.tensor(u_flat).unsqueeze(1) 
    
        # === 4️、 绘图 ===
    fig, ax = plt.subplots(figsize=(17, 9))
    im = ax.contourf(
        X, T, vu,
        levels=100, cmap='coolwarm', vmin=-1, vmax=1
    )
    ax.set_title(f"Ground Truth Burgers Solution (ν={nu:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()


    plt.savefig(f"burgers_gt_nu{nu:.5f}.png", dpi=300)
    print(f"✅ Figure saved as burgers_gt_nu{nu:.5}.png")

    plt.show()
    writer.add_figure("Ground Truth Burgers Solution", fig)

    return gt,X_test
