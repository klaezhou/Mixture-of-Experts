import torch
import argparse
from Equation.factory import get_equation # 建议增加一个工厂函数
from Pipeline.pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="PINNs Lightweight Solver")
    
    # --- basic config ---
    parser.add_argument("--eq", type=str, default="poisson", help="Equation name (poisson, burgers, etc.)")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # --- sampling ---
    parser.add_argument("--sample_method", type=str, default="grid")
    parser.add_argument("--Nf", type=int, default=10000)
    parser.add_argument("--Nb", type=int, default=200)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--n_per_edge", type=int, default=100)
    parser.add_argument("--w_pde", type=float, default=1.0)
    parser.add_argument("--w_bc", type=float, default=1.0)


    
    # --- training ---
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_scheduler", type=bool, default=True)
    parser.add_argument("--lr_step_size", type=int, default=1000)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--log_freq", type=int, default=500)
    
    return parser.parse_args()

def main():
    args = parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get the equation instance using the factory
    eq = get_equation(args)

    # Initialize the pipeline with the equation
    pipe = Pipeline(args=args, equation=eq)

    
    print(f"--- Starting {args.eq.upper()} train ---")
    pipe.trainer.train(pipe.data, epochs=args.iters) 

if __name__ == "__main__":
    main()