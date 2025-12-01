# ============================================================================
# Author: klae-zhou
# Date   : 2025-06-05
#  Description:
#   This is a demo for mixture of experts applied on function approximation.
# ============

import torch
from torch import nn
import torch.optim as optimi
import argparse
import numpy as np
from functorch import make_functional, vmap
from moe_module.moe import MOE_Model,MLP_Model
from moe_module.utils import parse_args, _init_data_dim1, get_optimizer, get_loss_fn, log_with_time,plot_dual_axis, plot_expert_useless_rank, get_activation, generate_data, get_writer,gates_experts_image, save_model, Timetxt, runcode
from moe_module.epi_rank import epi_rank_mlp
# download data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.autograd import grad
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def pde_residual(model, x_t, nu, moe_training=True):
    """
    u: model output
    x_t: tensor shape (N,2) with columns [x, t], requires_grad=True
    returns: residual r = u_t + u u_x - nu u_xx
    """
    x_t.requires_grad_(True)
    if moe_training:u, _ = model(x_t)
    else: u = model(x_t)
    g = grad(outputs=u,inputs=x_t,grad_outputs=torch.ones_like(u),create_graph=True,retain_graph=True)[0]                      # shape (N,2)
    u_x = g[:, 0:1]
    u_t = g[:, 1:2]
    u_xx = grad(outputs=u_x,inputs=x_t,grad_outputs=torch.ones_like(u_x),create_graph=True,retain_graph=True)[0][:, 0:1]
    r = u_t + u * u_x - nu * u_xx
    return r

def ground_truth(args):
    """
    u: model output
    x_t: tensor shape (N,2) with columns [x, t], requires_grad=True
    returns: residual r = u_t + u u_x - nu u_xx
    """
    gt,X_test=generate_data(args.vxn,args.vtn,args.nu, args.writer)
    X_test=X_test.to(args.device)
    gt=gt.to(args.device)
    args.gt=gt
    args.X_test=X_test

@log_with_time
def train_loop(X_init, X_bnd, X_f, X_total, u_init, model,loss_fn, optim, args,steps=100,moe_training=True):
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.lr_step, gamma=args.lr_decay)
    activation=get_activation(args.activation)
    aux_loss, init_loss, bnd_loss, f_loss, loss, total_loss = 0, 0, 0, 0, 0, 0
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
        nonlocal aux_loss, init_loss, bnd_loss, f_loss, loss, total_loss
        model.train()
        if moe_training:
            u_hat_init, _ = model(X_init)
            u_hat_bnd, _ = model(X_bnd)
            _, aux_loss = model(X_total)
        else:
            u_hat_init = model(X_init)
            u_hat_bnd = model(X_bnd)

        # calculate prediction loss
        init_loss = loss_fn(u_hat_init, u_init)
        bnd_loss = loss_fn(u_hat_bnd, torch.zeros_like(u_hat_bnd))  # zero boundary condition
        f_residual = pde_residual(model, X_f, args.nu, moe_training)
        f_loss = loss_fn(f_residual, torch.zeros_like(f_residual))
        loss = args.loss_coef_init *init_loss + args.loss_coef_bnd * bnd_loss + f_loss

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
            eval_model(step,X_init, X_bnd, X_f, X_total, u_init, model, loss_fn,moe_training,args)
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
                
                rank_mlp=epi_rank_mlp(model,args.x_interval,args.t_interval,args.x_integral_sample, args.t_integral_sample, args.epsilon, activation)
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
                rank_mlp=epi_rank_mlp(model,args.x_interval,args.t_interval,args.x_integral_sample, args.t_integral_sample,args.epsilon, activation ,moe_training=False,index=1)
                rank=rank_mlp.rank_mlp()
                total_rank_list.append(rank[args.plt_r])
                tqdm.write(f"Step {step+1}/{steps} - loss: {loss.item():.8f} -rank: {rank}")
                args.writer.add_scalar('MLP_Loss', total_loss.item(), step)

    if moe_training: save_model(model, "moe")
    else: save_model(model, "mlp")

    return model,total_loss_list,total_rank_list,total_useless_expert_rank
            
def eval_model(step,X_init, X_bnd, X_f, X_total, u_init, model, loss_fn,moe_training=True,args=None):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    if moe_training:
        u_hat_init, _ = model(X_init)
        u_hat_bnd, _ = model(X_bnd)
        u_total, aux_loss = model(X_total)
    else:
        u_hat_init = model(X_init)
        u_hat_bnd = model(X_bnd)
        u_total = model(X_total)

    init_loss = loss_fn(u_hat_init, u_init)
    bnd_loss = loss_fn(u_hat_bnd, torch.zeros_like(u_hat_bnd))  # zero boundary condition
    f_residual = pde_residual(model, X_f, args.nu, moe_training)
    f_loss = loss_fn(f_residual, torch.zeros_like(f_residual))
    loss = args.loss_coef_init *init_loss + args.loss_coef_bnd * bnd_loss + f_loss

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

# %%


def main():
    args=parse_args()
    torch.manual_seed(args.seed)
    get_writer(args)
    ground_truth(args)
    #init data and model
    data_X_init,data_X_bnd,data_X_f,data_X_total,data_u_init=_init_data_dim1(args.init_func,args.x_interval,args.t_interval,args.x_num_samples,args.t_num_samples,args.device)
    print(f"data_X_init shape: {data_X_init.shape},data_X_bnd shape: {data_X_bnd.shape},data_X_f shape: {data_X_f.shape},data_X_total shape: {data_X_total.shape},data_u_init shape: {data_u_init.shape} ")
    activation=get_activation(args.activation)
    loss_fn =get_loss_fn(args.lossfn)
    args.writer.add_text("Training arguments: ", 
                         f"Num Experts: {args.num_experts}, Hidden Size: {args.hidden_size}, Depth: {args.depth}, "
                         f"x Num Samples: {args.x_num_samples}, t Num Samples: {args.t_num_samples}, "
                         f"Coef_init: {args.loss_coef_init}, Coef_bnd: {args.loss_coef_bnd}, Coef_aux: {args.loss_coef}, "
                         f"Learning Rate: {args.lr}, Learning Rate Decay: {args.lr_decay}, Learning Rate Step: {args.lr_step}, "
                         f"Opt Steps: {args.opt_steps}, LBFGS Steps: {args.lbfgs_steps}, Seed: {args.seed}"
                         )

    model_moe=MOE_Model(args.input_size, args.num_experts,args.hidden_size,args.depth, args.output_size,args.k,args.loss_coef, activation).to(args.device)
    args.writer.add_text("Numbers of parameters: ",f"{model_moe._report_trainable()}")
    optimizer = get_optimizer(args.optim,model_moe.parameters(), lr=args.lr)
    model,total_loss_list_moe,rank_list_moe,total_useless_expert_rank_moe=train_loop(data_X_init,data_X_bnd,data_X_f,data_X_total,data_u_init, model_moe,loss_fn, optimizer, args,args.opt_steps, moe_training=True)
    eval_model(args.opt_steps+args.lbfgs_steps,data_X_init,data_X_bnd,data_X_f,data_X_total,data_u_init, model, loss_fn,moe_training=True,args=args)
    plot_dual_axis(np.array(total_loss_list_moe),np.array(rank_list_moe),args.opt_steps+args.lbfgs_steps,"moe",args.writer)
    plot_expert_useless_rank(np.array(total_loss_list_moe),np.array(total_useless_expert_rank_moe),args.opt_steps+args.lbfgs_steps,"moe",args.writer)

    # model_mlp=MLP_Model(args.input_size, args.hidden_size,args.depth, args.output_size, activation).to(args.device)
    # args.writer.add_text("Numbers of parameters: ", f"{model_mlp._report_trainable()}")
    # optimizer_mlp=get_optimizer(args.optim,model_mlp.parameters(), lr=args.lr)
    # model_mlp,total_loss_list_mlp,rank_list_mlp,total_useless_expert_rank_mlp=train_loop(data_X_init,data_X_bnd,data_X_f,data_X_total,data_u_init, model_mlp,loss_fn, optimizer_mlp, args,args.opt_steps,moe_training=False)
    # eval_model(args.opt_steps+args.lbfgs_steps,data_X_init,data_X_bnd,data_X_f,data_X_total,data_u_init, model_mlp, loss_fn,moe_training=False,args=args)
    # plot_dual_axis(np.array(total_loss_list_mlp),np.array(rank_list_mlp),args.opt_steps+args.lbfgs_steps,"mlp",args.writer)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()
