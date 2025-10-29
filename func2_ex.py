# -*- coding: utf-8 -*-
# Minimal PyTorch PINN for 1D Burgers: u_t + u u_x = nu u_xx,  x∈[-1,1], t∈[0,1]
# IC: u(x,0) = -sin(pi x),  BC: u(-1,t)=u(1,t)=0

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt

# ====== config ======
torch.set_default_dtype(torch.float64)  # PINN更稳
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

nu = 0.01 / math.pi  # 粘性
Nx_ic, Nt_ic = 256, 1     # 初值等距点数
Nt_bc = 256               # 边界等距点数（两侧各 Nt_bc）
Nf = 10000                # PDE内部点数（Sobol）
layers = [2, 50, 50, 50, 50, 1]
adam_steps = 30000         # 训练步数（可加大看看效果）

# ====== data: IC / BC / interior ======
# 初值：t=0, x in [-1,1]
x_ic = torch.linspace(-1.0, 1.0, Nx_ic, device=device).view(-1, 1)
t_ic = torch.zeros_like(x_ic, device=device)
u_ic = -torch.sin(math.pi * x_ic)  # u(x,0)

X_ic = torch.cat([x_ic, t_ic], dim=1)     # [Nx_ic,2]

# 边界：x=-1 和 x=1, t in [0,1]
t_bc = torch.linspace(0.0, 1.0, Nt_bc, device=device).view(-1, 1)
x_left  = -torch.ones_like(t_bc, device=device)
x_right =  torch.ones_like(t_bc,  device=device)

X_bc_left  = torch.cat([x_left,  t_bc], dim=1)
X_bc_right = torch.cat([x_right, t_bc], dim=1)
u_bc_left  = torch.zeros_like(t_bc, device=device)
u_bc_right = torch.zeros_like(t_bc, device=device)

# PDE 内部点：Sobol in (-1,1) × (0,1]
sobol = SobolEngine(dimension=2, scramble=True)
s = sobol.draw(Nf).to(device)
eps = 1e-7
x_f = -1.0 + 2.0 * (s[:, 0:1] * (1 - 2*eps) + eps)   # (-1,1)
t_f = (s[:, 1:2] * (1 - eps) + eps)                  # (0,1]
X_f = torch.cat([x_f, t_f], dim=1)

# ====== model ======
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-2):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.layers.append(nn.Tanh())
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        # 归一化边界，固定为 [x,t] ∈ [-1,1]×[0,1]
        self.register_buffer("lb", torch.tensor([-1.0, 0.0], dtype=torch.get_default_dtype()))
        self.register_buffer("ub", torch.tensor([ 1.0, 1.0], dtype=torch.get_default_dtype()))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 统一映射到 [-1,1]
        lb = self.lb.to(x.device, x.dtype); ub = self.ub.to(x.device, x.dtype)
        x = 2.0*(x - lb)/(ub - lb) - 1.0
        for layer in self.layers:
            x = layer(x)
        return x

model = MLP(layers).to(device)

# ====== PDE residual ======
def pde_residual(model, x_t, nu):
    x_t.requires_grad_(True)
    u = model(x_t)                      # (N,1)
    g = grad(u, x_t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]  # (N,2)
    u_x = g[:, 0:1]
    u_t = g[:, 1:2]
    u_xx = grad(u_x, x_t, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    return u_t + u*u_x - nu*u_xx

# ====== losses ======
mse = nn.MSELoss()

def loss_fn():
    # IC
    u_ic_pred = model(X_ic)
    loss_ic = mse(u_ic_pred, u_ic)
    # BC
    u_left_pred  = model(X_bc_left)
    u_right_pred = model(X_bc_right)
    loss_bc = mse(u_left_pred, u_bc_left) + mse(u_right_pred, u_bc_right)
    # PDE
    r = pde_residual(model, X_f, nu)
    loss_pde = mse(r, torch.zeros_like(r))
    # 总损失（简单相加，必要时可给 PDE 加权）
    return loss_ic + loss_bc + loss_pde, (loss_ic.item(), loss_bc.item(), loss_pde.item())

# ====== train (Adam 简版) ======
opt = torch.optim.Adam(model.parameters(), lr=8e-4)
for step in range(1, adam_steps+1):
    opt.zero_grad()
    loss, (lic, lbc, lpde) = loss_fn()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"[{step:5d}] loss={loss.item():.4e}  ic={lic:.2e}  bc={lbc:.2e}  pde={lpde:.2e}")

# ====== eval on grid and plot ======
with torch.no_grad():
    Nx_plot, Nt_plot = 200, 200
    xg = torch.linspace(-1.0, 1.0, Nx_plot, device=device)
    tg = torch.linspace(0.0, 1.0, Nt_plot, device=device)
    Xg, Tg = torch.meshgrid(xg, tg, indexing='ij')
    Xtest = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1)
    upred = model(Xtest).reshape(Nx_plot, Nt_plot).cpu().numpy()
    Xnp, Tnp = Xg.cpu().numpy(), Tg.cpu().numpy()

plt.figure(figsize=(10,5))
im = plt.contourf(Tnp, Xnp, upred, levels=100, cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('t'); plt.ylabel('x')
plt.title('PINN Prediction for Burgers (u(x,t))')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('burgers.png')
plt.show()

# ====== compare with ground truth (.mat) ======
import scipy.io
from scipy.interpolate import griddata

mat_path = "/home/zhy/Zhou/mixture_of_experts/data/burgers_shock.mat"
data = scipy.io.loadmat(mat_path)

# 数据集中：x,t 为列向量；usol 维度通常为 (Nx, Nt)，TF 版本里用了转置
t_mat = data["t"].flatten()[:, None]   # [Nt,1]
x_mat = data["x"].flatten()[:, None]   # [Nx,1]
U_exact = np.real(data["usol"]).T      # 变成 [Nt,Nx] -> 转置后配合下面的使用

# 用「数据集同样的网格」评估模型，避免插值误差
X_mat, T_mat = np.meshgrid(x_mat, t_mat)                 # X:[Nt,Nx], T:[Nt,Nx]
X_star = np.hstack((X_mat.reshape(-1,1), T_mat.reshape(-1,1)))  # [Nt*Nx,2]

with torch.no_grad():
    X_star_t = torch.tensor(X_star, dtype=torch.float64, device=device)
    U_pred_vec = model(X_star_t).reshape(-1, 1).detach().cpu().numpy()
    U_pred = U_pred_vec.reshape(len(t_mat), len(x_mat))  # [Nt,Nx]

# 误差（绝对+相对 L2）
abs_err = np.abs(U_pred - U_exact)                       # [Nt,Nx]
rel_l2 = np.linalg.norm(U_pred - U_exact) / np.linalg.norm(U_exact)
print(f"Relative L2 error vs GT: {rel_l2:.6e}")

# ====== 作图：预测 vs 真解 & 误差热图 ======
fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

# 左：真解
im0 = axs[0].contourf(t_mat.squeeze(), x_mat.squeeze(), U_exact.T, levels=100, cmap='coolwarm', vmin=-1, vmax=1)
axs[0].set_title("Ground Truth $u(x,t)$")
axs[0].set_xlabel("t"); axs[0].set_ylabel("x")
plt.colorbar(im0, ax=axs[0])

# 中：预测
im1 = axs[1].contourf(t_mat.squeeze(), x_mat.squeeze(), U_pred.T, levels=100, cmap='coolwarm', vmin=-1, vmax=1)
axs[1].set_title("PINN Prediction $\\hat{u}(x,t)$")
axs[1].set_xlabel("t"); axs[1].set_ylabel("x")
plt.colorbar(im1, ax=axs[1])

# 右：误差
im2 = axs[2].contourf(t_mat.squeeze(), x_mat.squeeze(), abs_err.T, levels=100, cmap='viridis')
axs[2].set_title(f"Abs Error $|\\hat{{u}}-u|$  (rel L2={rel_l2:.2e})")
axs[2].set_xlabel("t"); axs[2].set_ylabel("x")
plt.colorbar(im2, ax=axs[2])

plt.savefig("burgers_pred_vs_gt_and_error.png", dpi=300)
plt.show()

# ====== 切片对比：t=0.25, 0.50, 0.75 ======
def nearest_idx(arr, v):
    return int(np.argmin(np.abs(arr.squeeze() - v)))

t_slices = [0.25, 0.50, 0.75]
fig2, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
for j, tt in enumerate(t_slices):
    k = nearest_idx(t_mat, tt)
    axes[j].plot(x_mat, U_exact[k, :], 'b-',  lw=2, label='Exact')
    axes[j].plot(x_mat, U_pred[k, :],  'r--', lw=2, label='Pred')
    axes[j].set_title(f"t = {t_mat[k,0]:.2f}")
    axes[j].set_xlabel('x'); axes[j].set_ylabel('u')
    axes[j].set_xlim([x_mat.min(), x_mat.max()])
    axes[j].set_ylim([-1.1, 1.1])
    if j == 1:
        axes[j].legend()

plt.savefig("burgers_slices.png", dpi=300)
plt.show()