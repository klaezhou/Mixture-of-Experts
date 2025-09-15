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
    parser.add_argument("--device", type=str, default="cuda:3" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
    parser.add_argument("--input_size", type=int, default=1,help="Input size (funcion approximation x) ")
    parser.add_argument("--output_size", type=int, default=1,help="Output size (funcion approximation y=u(x)) ")
    parser.add_argument("--num_experts", type=int, default=20,help="Number of experts")
    parser.add_argument("--hidden_size", type=int, default=20,help="Hidden size of the MLP")
    parser.add_argument("--depth", type=int, default=4,help="Depth of the MOE model")
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--opt_steps", type=int, default=300)
    parser.add_argument("--function", type=str, default="cosx+sin100x+cos30x", help="function")
    parser.add_argument("--interval", type=str, default="[-1,1]")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--k", type=int, default=4,help="top-k selection")
    parser.add_argument("--loss_coef", type=float, default=1)
    parser.add_argument("--integral_sample", type=int, default=300, help="integral_sample")
    parser.add_argument("--plt_r", type=int, default=1)
    parser.add_argument("--decrease_rate", type=float, default=0.9)
    parser.add_argument("--index", type=int, default=0,help="index of the expert")
    return parser.parse_args()
def _init_data_dim1(func: str, interval: str, num_samples: int,device):
    """
    初始化数据 
    """
    interval = eval(interval)  # 例如 "[-1,1]" -> [-1, 1]
    # x = (interval[1] - interval[0]) * torch.rand(num_samples, 1) + interval[0]
    x=torch.linspace(interval[0], interval[1], num_samples).view(-1, 1)
    # 定义函数解析器
    def parse_function(expr: str):
        expr = expr.replace("cos30x", "np.cos(30*x_np)")
        expr = expr.replace("sin100x", "np.sin(100*x_np)")
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

def get_optimizer(name, model_params, lr=1e-3):
    name = name.lower()
    if name == "adam":
        return optim.Adam(model_params,lr=lr,   betas=(0.9, 0.999), eps=1e-9)
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

