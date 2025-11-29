import ast
import torch
from torch import nn
import torch.optim as optimi
from tqdm.auto import tqdm
import argparse
import numpy as np
from functorch import make_functional, vmap
from moe_module.moe import MOE_Model,MLP_Model
from moe_module.utils import get_optimizer, get_loss_fn, log_with_time,plot_dual_axis, plot_expert_useless_rank, get_activation, get_writer,gates_experts_image, save_model, Timetxt, runcode
from moe_module.epi_rank import epi_rank_mlp
from moe_module.tools import *
# download data
from torch.utils.data import DataLoader
from torch.autograd import grad
import matplotlib.pyplot as plt
import data.bg_gt as bg_gt
from torch.optim.lr_scheduler import StepLR

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Mixture-of-Experts model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--lr_decay", type=float, default=0.997, help="Learning rate decay factor.")
    parser.add_argument("--lr_step", type=int, default=50, help="Learning rate decay step size.")
    parser.add_argument("--device", type=str, default="cuda:5" if torch.cuda.is_available() else "cpu", help="Device to train on.")    
    parser.add_argument("--input_size", type=int, default=2,help="Input size (funcion approximation x) ")
    parser.add_argument("--output_size", type=int, default=1,help="Output size (funcion approximation y=u(x)) ")
    parser.add_argument("--num_experts", type=int, default=2,help="Number of experts")
    parser.add_argument("--hidden_size", type=int, default=16,help="Hidden size of each expert")
    parser.add_argument("--depth", type=int, default=7,help="Depth of the MOE model")
    parser.add_argument("--lossfn", type=str, default="mse", help="Loss function.")
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--opt_steps", type=int, default=10000, help="number of steps for AdamW")
    parser.add_argument("--lbfgs_steps",type=int,default=10000,help="number of steps for lbfgs")
    parser.add_argument("--activation", type=str, default="tanh", help="activation_function")
    parser.add_argument("--x_interval", type=str, default="[-1,1]")
    parser.add_argument("--y_interval", type=str, default="[-1,1]")
    parser.add_argument("--x_num_samples", type=int, default=200)
    parser.add_argument("--y_num_samples", type=int, default=200)
    parser.add_argument("--k", type=int, default=1,help="top-k selection")
    parser.add_argument("--loss_coef", type=float, default=1)
    parser.add_argument("--x_integral_sample", type=int, default=200, help="x_integral_sample")
    parser.add_argument("--y_integral_sample", type=int, default=200, help="t_integral_sample")
    parser.add_argument("--a", type=int, default=1, help="coefficient a in pde \Delta u + \pi^2 * (a^2 + b^2)sin(a*\pi*x)sin(b*\pi*y) = 0")
    parser.add_argument("--b", type=int, default=5, help="coefficient b in pde \Delta u + \pi^2 * (a^2 + b^2)sin(a*\pi*x)sin(b*\pi*y) = 0")
    parser.add_argument("--plt_r", type=int, default=-1)
    parser.add_argument("--epsilon", type=float, default=1e-3,help="epsilon for rank")
    parser.add_argument("--loss_coef_bnd", type=float, default=5)
    parser.add_argument("--vtn",type=int,default=200)
    parser.add_argument("--vxn",type=int,default=200)
    parser.add_argument("--gt",type=torch.Tensor,default=None,help="ground truth")
    parser.add_argument("--X_test",type=torch.Tensor,default=None,help="test data")
    parser.add_argument("--seed",type=int,default=1234) #1234
    parser.add_argument("--smooth_steps",type=int,default=5,help="number of steps for smooth mode")
    parser.add_argument("--smooth_lb",type=int,default=5000,help="number lower bound of steps for smooth mode")
    parser.add_argument("--x_integral_interval" ,type=str, default="[-1,1]")
    parser.add_argument("--y_integral_interval", type=str, default="[-1,1]")
    return parser.parse_args()

def pde_residual(model, x_y, a, b, moe_training=True):
    x_y.requires_grad_(True)
    if moe_training:u, _ = model(x_y)
    else: u = model(x_y)
    g = grad(outputs=u,inputs=x_y,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]                      # shape (N,2)
    u_x = g[:, 0:1]
    u_y = g[:, 1:2]

    # 二阶: u_xx = d/dx(u_x)
    u_xx = grad(
        outputs=u_x,inputs=x_y,grad_outputs=torch.ones_like(u_x),create_graph=True,retain_graph=True)[0][:, 0:1]
    u_yy = grad(
        outputs=u_y,inputs=x_y,grad_outputs=torch.ones_like(u_x),create_graph=True,retain_graph=True)[0][:, 1:2]

    r = u_xx + u_yy + (a**2 + b**2) * np.pi**2 * torch.sin(a * np.pi * x_y[:, 0:1]) * torch.sin(b * np.pi * x_y[:, 1:2])
    return r

def ground_truth(args):
    x_lo, x_hi = ast.literal_eval(args.x_interval)
    y_lo, y_hi = ast.literal_eval(args.y_interval)
    vx_torch = torch.tensor(np.linspace(x_lo, x_hi, args.vxn), dtype=torch.get_default_dtype()).view(-1, 1)
    vy_torch = torch.tensor(np.linspace(y_lo, y_hi, args.vtn), dtype=torch.get_default_dtype()).view(-1, 1)
    X, Y = torch.meshgrid(vx_torch.squeeze(), vy_torch.squeeze(), indexing='ij')
    X_test = torch.stack([X.flatten(), Y.flatten()], dim=1)

    u_sel = torch.sin(args.a * np.pi * X) * torch.sin(args.b * np.pi * Y)
    gt = torch.tensor(u_sel.flatten(), dtype=torch.get_default_dtype()).view(-1, 1)

    fig, ax = plt.subplots(figsize=(17, 9))
    im = ax.contourf(
        X, Y, u_sel,
        levels=100, cmap='coolwarm', vmin=-1, vmax=1
    )
    ax.set_title(f"Ground Truth Possion Solution (a={args.a}, b={args.b})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(f"Possion_gt_a{args.a}_b{args.b}.png", dpi=300)
    plt.show()
    args.writer.add_figure("Ground Truth", fig)

    X_test=X_test.to(args.device)
    gt=gt.to(args.device)
    args.gt=gt
    args.X_test=X_test

def _init_data_dim1(x_interval: str, y_interval: str, x_num_samples: int, y_num_samples: int, device):
    """
    初始化数据 
    """
    target = x_num_samples * y_num_samples
    x_lo, x_hi = ast.literal_eval(x_interval)
    y_lo, y_hi = ast.literal_eval(y_interval)

    torch.set_default_dtype(torch.float64)
    device = torch.device(device)
    # boundary
    # 等距网格（用于初值与边界）
    x_lin = torch.linspace(x_lo, x_hi, x_num_samples*3, device=device, dtype=torch.get_default_dtype()).view(-1, 1)  # [Nx,1]
    y_lin = torch.linspace(y_lo, y_hi, y_num_samples*3, device=device, dtype=torch.get_default_dtype()).view(-1, 1)  # [Ny,1]

    #  初值面：y = y0（等距 x）
    ymin, ymax = y_lin[0, 0], y_lin[-1, 0]
    X_bnd_down = torch.stack([x_lin.squeeze(1), torch.full_like(x_lin.squeeze(1), ymin)], dim=1)  # [Nx,2]
    X_bnd_up = torch.stack([x_lin.squeeze(1), torch.full_like(x_lin.squeeze(1), ymax)], dim=1)  # [Nx,2]
    xmin, xmax = x_lin[0, 0], x_lin[-1, 0]
    X_bnd_left  = torch.stack([torch.full_like(y_lin.squeeze(1), xmin), y_lin.squeeze(1)], dim=1)  # [Ny,2]
    X_bnd_right = torch.stack([torch.full_like(y_lin.squeeze(1), xmax), y_lin.squeeze(1)], dim=1)  # [Ny,2]
    X_bnd = torch.cat([X_bnd_down, X_bnd_up, X_bnd_left, X_bnd_right], dim=0)  # [2*NX+2*Ny,2]
    # Interior

    ############################
    #######euqi-steps############
    x_lin = torch.linspace(x_lo, x_hi, (x_num_samples+2)*2, device=device, dtype=torch.get_default_dtype()).view(-1, 1)
    y_lin = torch.linspace(y_lo, y_hi, (y_num_samples+2)*2, device=device, dtype=torch.get_default_dtype()).view(-1, 1)
    

    if x_num_samples >= 3 and y_num_samples >= 2:
        # 内部 x：去掉两端；内部 y：去掉 y0
        x_in = x_lin[1:-1].squeeze(1)   # [Nx-2]
        y_in = y_lin[1:  ].squeeze(1)   # [Ny-1]

        # 生成网格并展平为 [n_f, 2]
        Xg, Yg = torch.meshgrid(x_in, y_in, indexing='ij')   # X 优先
        X_f = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)  # [(Nx-2)(Ny-1), 2]
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
    X_total = torch.cat([X_bnd, X_f], dim=0)
    X_total = torch.unique(X_total, dim=0)

    X_bnd = X_bnd.to(device)
    X_f = X_f.to(device)
    X_total = X_total.to(device)
    return X_bnd, X_f, X_total


@log_with_time
def train_loop(X_bnd, X_f, X_total, model,loss_fn, optim, args,steps=100,moe_training=True):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_step, gamma=args.lr_decay)
    activation=get_activation(args.activation)
    aux_loss, bnd_loss, f_loss, loss, total_loss = 0, 0, 0, 0, 0
    total_loss_list, total_rank_list, total_useless_expert_rank=[],[],[]
    tqdm_range=tqdm(range(steps+args.lbfgs_steps), desc=f"Training with code {runcode}-{Timetxt}")
    optimizer = optimi.LBFGS(
                model.parameters(),lr=0.5, max_iter=20,        # 每次step内部最多迭代次数 
                max_eval=None,      # 评估上限（默认= max_iter*1.25）
                tolerance_grad=1e-13,
                tolerance_change=1e-13,
                history_size=50,   # 保留的校正对数量
                line_search_fn="strong_wolfe",  # 只有这个被支持
            )
    def compute_loss_and_grad():
        nonlocal aux_loss, bnd_loss, f_loss, loss, total_loss
        model.train()
        if moe_training:
            u_hat_bnd, _ = model(X_bnd)
            _, aux_loss = model(X_total)
        else:
            u_hat_bnd = model(X_bnd)

        # calculate prediction loss
        bnd_loss = loss_fn(u_hat_bnd, torch.zeros_like(u_hat_bnd))  # zero boundary condition
        f_residual = pde_residual(model, X_f, args.a, args.b, moe_training)
        f_loss = loss_fn(f_residual, torch.zeros_like(f_residual))
        loss = args.loss_coef_bnd * bnd_loss + f_loss

        # combine losses
        if moe_training: total_loss = loss + aux_loss
        else: total_loss = loss
        total_loss.backward()
        return total_loss
    
    
    step_count=args.smooth_steps
    
    for step in tqdm_range:
        model.train()
        
        if moe_training :
            step_count -=1
            if model.moe.smooth and step_count<=0:
                model.moe.smoothing(step,args.smooth_lb)
                step_count=args.smooth_steps
            elif step_count<=0 :
                model.moe.smoothing(step,args.smooth_lb)
                step_count=args.smooth_steps

        if step % 100 == 0:
            eval_model(step, X_bnd, X_f, X_total, model, loss_fn,moe_training,args)
            if moe_training:
                gates_experts_image(model,args.X_test, step, args.writer)

            #update points
            # args.seed=args.seed+step
            # torch.manual_seed(args.seed)
            # X_init,X_bnd,X_f,X_total,u_init=_init_data_dim1(args.init_func,args.x_interval,args.t_interval,args.x_num_samples,args.t_num_samples,args.device,args)
        
        if step > args.opt_steps:
            def closure():
                optimizer.zero_grad()
                total_loss = compute_loss_and_grad()
                return total_loss
            optimizer.step(closure)
        else:
            optim.zero_grad()
            total_loss = compute_loss_and_grad()
            optim.step()
            scheduler.step()
            
        total_loss_list.append(total_loss.item())

        if moe_training:
            tqdm_range.set_postfix({'loss': f'{loss.item():.8f}', 'aux_loss': f'{aux_loss.item():.8f}'})
        else:
            tqdm_range.set_postfix({'loss': f'{loss.item():.8f}'})

        if step % 100 == 0 or step == steps - 1:
            if moe_training:
                # rank_moe=epi_rank_moe(model,args.interval,args.integral_sample)
                # rank=rank_moe.rank_moe()
                
                rank_mlp=epi_rank_mlp(model,args.x_interval,args.y_interval,args.x_integral_sample, args.y_integral_sample, args.epsilon, activation)
                rank_list=rank_mlp.rank_mlp()
                total_rank_list.append(rank_list[0])
                rank_list_experts=rank_mlp.experts_rank_mlp()
                tqdm.write(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -aux_loss: {aux_loss.item():.8f} -rank: {rank_list[0]},{rank_list} -experts_rank: {rank_list_experts[:-2]} -total_experts_rank: {rank_list_experts[-2]} -useless_expert_rank: {rank_list_experts[-1]}")
                args.writer.add_scalar('MoE_Loss', total_loss.item(), step)
                args.writer.add_scalar('Aux_Loss', aux_loss.item(), step)
                args.writer.add_scalar('MoE_Rank', rank_list[0], step)
                args.writer.add_scalar('Useless_Expert_Rank', rank_list_experts[-1], step)
                total_useless_expert_rank.append(rank_list_experts[-1])
            else:
                rank_mlp=epi_rank_mlp(model,args.x_interval,args.y_interval,args.x_integral_sample, args.y_integral_sample,args.epsilon, activation ,moe_training=False,index=1)
                rank=rank_mlp.rank_mlp()
                total_rank_list.append(rank[args.plt_r])
                tqdm.write(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -rank: {rank}")
                args.writer.add_scalar('MLP_Loss', total_loss.item(), step)

    if moe_training: save_model(model, "moe")
    else: save_model(model, "mlp")

    return model,total_loss_list,total_rank_list,total_useless_expert_rank
            
def eval_model(step, X_bnd, X_f, X_total, model, loss_fn,moe_training=True,args=None):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    if moe_training:
        u_hat_bnd, _ = model(X_bnd)
        u_total, aux_loss = model(X_total)
    else:
        u_hat_bnd = model(X_bnd)
        u_total = model(X_total)

    bnd_loss = loss_fn(u_hat_bnd, torch.zeros_like(u_hat_bnd))  # zero boundary condition
    f_residual = pde_residual(model, X_f, args.a, args.b, moe_training)
    f_loss = loss_fn(f_residual, torch.zeros_like(f_residual))
    loss = args.loss_coef_bnd * bnd_loss + f_loss

    if moe_training:
        tqdm.write("MoE_Model Evaluation Results - loss: {:.8f}, aux_loss: {:.8f}".format(loss.item(), aux_loss.item()))
    else:
        tqdm.write("MLP_Model Evaluation Results - loss: {:.8f}".format(loss.item()))


    # 1---------- 绘制预测----------
    if moe_training:
        u_pred,_= model(args.X_test)
    else:
        u_pred = model(args.X_test)
    x = args.X_test[:, 0].detach().cpu().numpy()
    t = args.X_test[:, 1].detach().cpu().numpy()
    u_pred = u_pred.detach().cpu().numpy().flatten()

    # 尝试重建网格（假设 x 是 meshgrid flatten 后的）
    nx = len(np.unique(x))
    nt = len(np.unique(t))
    X = np.unique(x)
    T = np.unique(t)
    X, T = np.meshgrid(X, T, indexing='ij')
    u_pred = u_pred.reshape(nx, nt)
    
    fig, axs = plt.subplots(figsize=(17, 9))

    im0 = axs.contourf(X, T, u_pred, levels=100, cmap='coolwarm', vmin=-1, vmax=1)
    axs.set_title("Solution_pred")
    axs.set_xlabel("x")
    axs.set_ylabel("t")
    fig.colorbar(im0, ax=axs)

    fig.tight_layout()
    if moe_training:
        plt.savefig("solution_pred_moe.png")
        args.writer.add_figure("Solution_pred_MoE", fig, step)
    else:
        plt.savefig("solution_pred_mlp.png")
        args.writer.add_figure("Solution_pred_MLP", fig, step)
    plt.show()
    
    #  2----------绘制误差-----------
    if moe_training:
        u_test,_= model(args.X_test)
        u_error = torch.abs(u_test - args.gt)
    else:
        u_error = torch.abs(model(args.X_test) - args.gt)
    rell2=torch.norm(u_error, p=2) / torch.norm(args.gt, p=2)
    tqdm.write(f"L2 absolute error: {rell2.item():.8f}")
    if moe_training:
        # np.save("l2_moe.npy", np.array(rell2.detach().cpu().numpy()))
        args.writer.add_scalar('MoE_L2_Error', rell2.item(), step)
    else:
        # np.save("l2_mlp.npy", np.array(rell2.detach().cpu().numpy()))
        args.writer.add_scalar('MLP_L2_Error', rell2.item(), step)
    

    
    x = args.X_test[:, 0].detach().cpu().numpy()
    t = args.X_test[:, 1].detach().cpu().numpy()
    u_error = u_error.detach().cpu().numpy().flatten()

    # 尝试重建网格（假设 x 是 meshgrid flatten 后的）
    nx = len(np.unique(x))
    nt = len(np.unique(t))
    X = np.unique(x)
    T = np.unique(t)
    X, T = np.meshgrid(X, T, indexing='ij')
    u_error = u_error.reshape(nx, nt)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    im2 = ax2.contourf(X, T, u_error, levels=100, cmap='viridis')
    ax2.set_title("Absolute Error")
    ax2.set_xlabel("x")   
    ax2.set_ylabel("t")
    fig2.colorbar(im2, ax=ax2)

    fig2.tight_layout()
    if moe_training:
        plt.savefig("abe_moe.png")
        args.writer.add_figure("Absolute_Error_MoE", fig2, step)
    else:
        plt.savefig("abe_mlp.png")
        args.writer.add_figure("Absolute_Error_MLP", fig2, step)
    plt.show()

















def main():
    args=parse_args()
    torch.manual_seed(args.seed)
    get_writer(args)
    ground_truth(args)
    #init data and model
    data_X_bnd,data_X_f,data_X_total=_init_data_dim1(args.x_interval,args.y_interval,args.x_num_samples,args.y_num_samples,args.device)
    activation=get_activation(args.activation)
    loss_fn =get_loss_fn(args.lossfn)
    args.writer.add_text("Training arguments: ", 
                         f"Num Experts: {args.num_experts}, Hidden Size: {args.hidden_size}, Depth: {args.depth}, "
                         f"x Num Samples: {args.x_num_samples}, y Num Samples: {args.y_num_samples}, "
                         f"Coef_bnd: {args.loss_coef_bnd}, Coef_aux: {args.loss_coef}, "
                         f"Learning Rate: {args.lr}, Learning Rate Decay: {args.lr_decay}, Learning Rate Step: {args.lr_step}, "
                         f"Opt Steps: {args.opt_steps}, LBFGS Steps: {args.lbfgs_steps}, Seed: {args.seed}"
                         )

    model_moe=MOE_Model(args.input_size, args.num_experts,args.hidden_size,args.depth, args.output_size,args.k,args.loss_coef, activation).to(args.device)
    args.writer.add_text("Numbers of parameters: ",f"{model_moe._report_trainable()}")
    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model,total_loss_list_moe,rank_list_moe,total_useless_expert_rank_moe=train_loop(data_X_bnd,data_X_f,data_X_total, model_moe,loss_fn, optimizer, args,args.opt_steps, moe_training=True)
    eval_model(args.opt_steps+args.lbfgs_steps,data_X_bnd,data_X_f,data_X_total, model, loss_fn,moe_training=True,args=args)
    plot_dual_axis(np.array(total_loss_list_moe),np.array(rank_list_moe),args.opt_steps+args.lbfgs_steps,"moe",args.writer)
    plot_expert_useless_rank(np.array(total_loss_list_moe),np.array(total_useless_expert_rank_moe),args.opt_steps+args.lbfgs_steps,"moe")

    # model_mlp=MLP_Model(args.input_size, args.hidden_size,args.depth, args.output_size, activation).to(args.device)
    # args.writer.add_text("Numbers of parameters: ", f"{model_mlp._report_trainable()}")
    # optimizer_mlp=get_optimizer(args.optim,model_mlp.parameters(), lr=args.lr)
    # model_mlp,total_loss_list_mlp,rank_list_mlp,total_useless_expert_rank_mlp=train_loop(data_X_bnd,data_X_f,data_X_total, model_mlp,loss_fn, optimizer_mlp, args,args.opt_steps,moe_training=False)
    # eval_model(args.opt_steps+args.lbfgs_steps,data_X_bnd,data_X_f,data_X_total, model_mlp, loss_fn,moe_training=False,args=args)
    # plot_dual_axis(np.array(total_loss_list_mlp),np.array(rank_list_mlp),args.opt_steps+args.lbfgs_steps,"mlp",args.writer)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()