# viz/rank_callback.py
import os
from viz.callbacks import Callback
from metrics.epsilon_rank2D import epsilon_rank_model_2d_trapz
import matplotlib.pyplot as plt
import numpy as np

class RankCallback(Callback):
    def __init__(self, args, equation, feature_getter, freq_dict={"adam":50,"lbfgs":5}):
        super().__init__()
        self.args = args
        self.eq = equation
        self.feature_getter = feature_getter
        self.nx= getattr(args, "int_grid_n", 200)
        self.ny= getattr(args, "int_grid_n", 200)
        self.lowx= getattr(args, "int_domain_lowx", -1.0)
        self.highx= getattr(args, "int_domain_highx", 1.0)
        self.lowy= getattr(args, "int_domain_lowy", -1.0)
        self.highy= getattr(args, "int_domain_highy", 1.0)
        self.eps= getattr(args, "eps", 1e-3)     # lambda > eps 
        self.freq_dict = freq_dict
        self.freq = None
        self.save_dir = args.save_dir
        self.iters = []
        self.ranks = []
        os.makedirs(self.save_dir, exist_ok=True)

    def on_iter_end(self, trainer, it: int, loss_dict: dict):
        
        if it % self.freq != 0:
            return

        r, evals = epsilon_rank_model_2d_trapz(trainer.model, self.feature_getter,self.nx,self.ny,self.lowx,self.highx,self.lowy,self.highy, eps=self.eps)
        # features_getter: model, x_grid -> Phi [N,m]

        self.iters.append(it)
        self.ranks.append(r)
        self._save_plot(it)
        self._save_distribution(it, evals)

        print(f"[Rank] iter={it} rank={r}")
    
    def on_train_end(self, trainer):
        r, evals = epsilon_rank_model_2d_trapz(trainer.model, self.feature_getter,self.nx,self.ny,self.lowx,self.highx,self.lowy,self.highy, eps=self.eps)
        # features_getter: model, x_grid -> Phi [N,m]

        it=self.iters[-1]+1
        self.iters.append(it)
        self.ranks.append(r)
        self._save_plot(it)
        self._save_distribution(it, evals)

        print(f"[Rank] iter={it} rank={r}")

    def _save_plot(self, it: int):
        plt.figure()
        plt.plot(self.iters, self.ranks, label="eps-rank")
        plt.xlabel("Iteration")
        plt.ylabel("Rank")
        plt.title(f"Epsilon-Rank Curve (iter={it})")
        plt.tight_layout()

        out_path = os.path.join(self.save_dir, "rank_curve.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_distribution(self, it: int, evals):


        # evals: torch tensor [m]
        e = evals.detach().cpu().numpy()
        e = np.sort(e)

        plt.figure()
        plt.plot(e, marker="o", linewidth=1)
        plt.axhline(self.eps, linestyle="--", label=f"eps={self.eps}")

        plt.yscale("log")  # usually eigenvalues span many orders
        plt.xlabel("Index (sorted)")
        plt.ylabel("Eigenvalue")
        plt.title(f"Eigenvalue Spectrum (iter={it})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(self.save_dir, "rank_distribution.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

    def on_phase_begin(self, trainer, phase: str):
        if phase not in ("adam", "lbfgs"):
            print(f"[Warning] Unknown phase '{phase}', using base freq.")
        self.freq = self.freq_dict.get(phase, self._base_freq)