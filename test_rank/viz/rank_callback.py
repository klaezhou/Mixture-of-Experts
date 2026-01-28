# viz/rank_callback.py
import os
from viz.callbacks import Callback
from metrics.epsilon_rank2D import epsilon_rank_model_2d_trapz
import matplotlib.pyplot as plt

class RankCallback(Callback):
    def __init__(self, args, equation, feature_getter, freq=500):
        self.args = args
        self.eq = equation
        self.feature_getter = feature_getter
        self.nx= getattr(args, "int_grid_n", 200)
        self.ny= getattr(args, "int_grid_n", 200)
        self.low= getattr(args, "int_domain_low", -1.0)
        self.high= getattr(args, "int_domain_high", 1.0)
        self.eps= getattr(args, "eps", 1e-3)     # lambda > eps 
        self.freq = freq
        self.save_dir = args.save_dir
        self.iters = []
        self.ranks = []
        os.makedirs(self.save_dir, exist_ok=True)

    def on_iter_end(self, trainer, it: int, loss_dict: dict):
        if it % self.freq != 0:
            return

        r, evals = epsilon_rank_model_2d_trapz(trainer.model, self.feature_getter,self.nx,self.ny,self.low,self.high, eps=self.eps)
        # features_getter: model, x_grid -> Phi [N,m]


        self.iters.append(it)
        self.ranks.append(r)
        self._save_plot(it)

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
