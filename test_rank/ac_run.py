import torch
import argparse
from Equation.factory import get_equation 
from Pipeline.pipeline import Pipeline
from viz.callbacks import Callback
from viz.error_plot_callback import ErrorPlotCallback
from viz.loss_plot_callback import LossPlotCallback
from viz.checkpoint_callback import CheckpointCallback
from viz.rank_callback import RankCallback
from model.factory import get_feature_getter
def parse_args():
    parser = argparse.ArgumentParser(description="PINNs Lightweight Solver")
    
    # --- basic config ---
    parser.add_argument("--eq", type=str, default="ac", help="Equation name (poisson, ac, etc.)")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--model", type=str, default="moe_d", help="Model architecture (moe_d, mlp etc.)")
    parser.add_argument("--seed", type=int, default=2026)
    
    # --- sampling ---
    parser.add_argument("--mode_n", type=float, default=5.0)
    
    parser.add_argument("--sample_method", type=str, default="grid") # random or grid
    parser.add_argument("--Nf", type=int, default=10000)
    parser.add_argument("--Nb", type=int, default=1000)
    parser.add_argument("--Ni", type=int, default=1000)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--nt", type=int, default=100)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=20.0)
    parser.add_argument("--w_ic", type=float, default=20.0)


    # --- modeling ---
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--gating_hidden_size", type=int, default=4)
    parser.add_argument("--gating_depth", type=int, default=1)
    parser.add_argument("--use_double", action="store_true")
    
    # --- training ---
    parser.add_argument("--iters", type=int, default=1000)  # adam iters
    parser.add_argument("--use_lbfgs", action="store_true") # use lbfgs or not
    parser.add_argument("--lbfgs_iter", type=int, default=700) # lbfgs iters
    parser.add_argument("--lbfgs_lr", type=float, default=1.0) # lbfgs lr
    parser.add_argument("--lr", type=float, default=8e-3) # adam lr
    parser.add_argument("--use_scheduler", type=bool, default=True)
    parser.add_argument("--lr_step_size", type=int, default=2000)
    parser.add_argument("--lr_gamma", type=float, default=0.8) 
    parser.add_argument("--log_freq", type=dict, default={"adam": 500, "lbfgs": 10}) # in trainer.py for print loss
    parser.add_argument("--checkpoint_freq", type=dict, default={"adam": 10000, "lbfgs": 200})

    
    # --- plot ---
    parser.add_argument("--save_dir", type=str, default="/home/zhy/Zhou/mixture_of_experts/test_rank/outputs/ac")
    parser.add_argument("--plot_freq", type=dict, default={"adam": 500, "lbfgs": 50} ) # in error_plot_callback.py for plot error
    parser.add_argument("--loss_freq", type=dict, default={"adam": 5, "lbfgs": 1}) # in checkpoint_callback.py for saving model
    parser.add_argument("--eval_grid_nx", type=int, default=240)
    parser.add_argument("--eval_grid_nt", type=int, default=120)
    
    # --- rank callback ---
    parser.add_argument("--rank_freq", type=dict, default={"adam": 50, "lbfgs": 30}) 
    parser.add_argument("--int_grid_n", type=int, default=200)
    parser.add_argument("--int_domain_lowx", type=float, default=0.0)  #time
    parser.add_argument("--int_domain_highx", type=float, default=1.0)
    parser.add_argument("--int_domain_lowy", type=float, default=-1.0)  #x
    parser.add_argument("--int_domain_highy", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-3)
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get the equation instance using the factory
    eq = get_equation(args)

    # Initialize the pipeline with the equation
    
    err_cb = ErrorPlotCallback(args=args, equation=eq,freq_dict=args.plot_freq)
    loss_cb=LossPlotCallback(args=args,freq_dict=args.loss_freq)
    check_cb = CheckpointCallback(args=args, freq_dict=args.checkpoint_freq)
    feature_getter= get_feature_getter(args)
    rank_cb=    RankCallback(args=args, equation=eq, feature_getter=feature_getter, freq_dict=args.rank_freq)
    callbacks = [err_cb, loss_cb, check_cb,rank_cb]
    pipe = Pipeline(args=args, equation=eq, callbacks=callbacks)
    
    print(f"--- Starting {args.eq.upper()} train ---")
    pipe.trainer.train_adam(pipe.data)
    pipe.trainer.train_lbfgs(pipe.data)
    print(f"--- {args.eq.upper()} train finished ---")
    


main()