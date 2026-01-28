import torch
import argparse
from Equation.factory import get_equation 
from Pipeline.pipeline import Pipeline
from viz.callbacks import Callback
from viz.error_plot_callback import ErrorPlotCallback
from viz.loss_plot_callback import LossPlotCallback
from viz.checkpoint_callback import CheckpointCallback
from viz.rank_callback import RankCallback
from model.moe_d import moe_penultimate_getter
def parse_args():
    parser = argparse.ArgumentParser(description="PINNs Lightweight Solver")
    
    # --- basic config ---
    parser.add_argument("--eq", type=str, default="poisson", help="Equation name (poisson, burgers, etc.)")
    parser.add_argument("--device", type=str, default="cuda:5")
    
    # --- sampling ---
    parser.add_argument("--sample_method", type=str, default="random") # random or grid
    parser.add_argument("--Nf", type=int, default=10000)
    parser.add_argument("--Nb", type=int, default=200)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--n_per_edge", type=int, default=100)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=1.0)

    # --- modeling ---
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=20)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--gating_hidden_size", type=int, default=4)
    parser.add_argument("--gating_depth", type=int, default=1)
    parser.add_argument("--use_double", action="store_true")
    
    # --- training ---
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--use_scheduler", type=bool, default=True)
    parser.add_argument("--lr_step_size", type=int, default=2000)
    parser.add_argument("--lr_gamma", type=float, default=0.8) 
    parser.add_argument("--log_freq", type=int, default=500) # in trainer.py for print loss

    
    # --- plot ---
    parser.add_argument("--save_dir", type=str, default="/home/zhy/Zhou/mixture_of_experts/test_rank/outputs/possion")
    parser.add_argument("--plot_freq", type=int, default=3000) # in error_plot_callback.py for plot error
    parser.add_argument("--loss_freq", type=int, default=50) # in checkpoint_callback.py for saving model
    parser.add_argument("--eval_grid_n", type=int, default=200)
    parser.add_argument("--domain_low", type=float, default=-1.0)
    parser.add_argument("--domain_high", type=float, default=1.0)
    
    # --- rank callback ---
    parser.add_argument("--rank_freq", type=int, default=300) 
    parser.add_argument("--int_grid_n", type=int, default=200)
    parser.add_argument("--int_domain_low", type=float, default=-1.0)
    parser.add_argument("--int_domain_high", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-3)
    
    
    
    
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get the equation instance using the factory
    eq = get_equation(args)

    # Initialize the pipeline with the equation

    err_cb = ErrorPlotCallback(args=args, equation=eq,freq=args.plot_freq)
    loss_cb=LossPlotCallback(args=args,freq=args.loss_freq)
    check_cb = CheckpointCallback(args=args, freq=10000)
    rank_cb=    RankCallback(args=args, equation=eq, feature_getter=moe_penultimate_getter, freq=args.rank_freq)
    callbacks = [err_cb, loss_cb, check_cb,rank_cb]
    pipe = Pipeline(args=args, equation=eq, callbacks=callbacks)
    
    print(f"--- Starting {args.eq.upper()} train ---")
    pipe.trainer.train(pipe.data, epochs=args.iters)
    

if __name__ == "__main__":
    main()