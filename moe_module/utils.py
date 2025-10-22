import logging
import time
from functools import wraps
import argparse
import torch
from torch import nn
import torch.optim as optim
import  numpy as np
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
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
    parser.add_argument("--nu", type=float, default=0.01,help="nu in equation u_t + u*u_x - nu*u_xx=0")
    parser.add_argument("--plt_r", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--loss_coef_init", type=float, default=1)
    parser.add_argument("--loss_coef_bnd", type=float, default=1)
    return parser.parse_args()
def _init_data_dim1(func: str, x_interval: str, t_interval: str, x_num_samples: int, t_num_samples: int, device):
    """
    初始化数据 
    """
    x_interval = eval(x_interval)  # 例如 "[-1,1]" -> [-1, 1]
    t_interval = eval(t_interval)
    # x = (interval[1] - interval[0]) * torch.rand(num_samples, 1) + interval[0]
    x=torch.linspace(x_interval[0], x_interval[1], x_num_samples).view(-1, 1)
    t=torch.linspace(t_interval[0], t_interval[1], t_num_samples).view(-1, 1)
    # 定义函数解析器
    def parse_function(expr: str):
        expr = expr.replace("-sin(pi*x)", "-np.sin(np.pi*x_np)")
        def f(x_tensor):
            x_np = x_tensor.numpy()
            y_np = eval(expr)
            return torch.from_numpy(y_np)
        return f

    f = parse_function(func)
    u_init = f(x)

    X, T = torch.meshgrid(x.squeeze(), t[0].squeeze(), indexing='ij')
    X_init = torch.stack([X.flatten(), T.flatten()], dim=1)
    X_init = X_init.to(device)

    X, T = torch.meshgrid(x[[0,-1]].squeeze(), t[1:].squeeze(), indexing='ij')
    X_bnd = torch.stack([X.flatten(), T.flatten()], dim=1)
    X_bnd = X_bnd.to(device)

    X, T = torch.meshgrid(x[1:-1].squeeze(), t[1:].squeeze(), indexing='ij')
    X_f = torch.stack([X.flatten(), T.flatten()], dim=1)
    X_f = X_f.to(device)

    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    X_total = torch.stack([X.flatten(), T.flatten()], dim=1)
    X_total = X_total.to(device)

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