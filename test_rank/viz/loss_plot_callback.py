# viz/loss_plot_callback.py
import os
import matplotlib.pyplot as plt
from viz.callbacks import Callback

class LossPlotCallback(Callback):
    def __init__(self, args,  freq=500):
        self.args = args
        self.save_dir =  args.save_dir
        self.freq = freq
        os.makedirs(self.save_dir, exist_ok=True)

        self.iters = []
        self.history = {}  # key -> list of floats

    def on_iter_end(self, trainer, it: int, loss_dict: dict):
        # loss_dict should already be pure floats in your trainer; if not, it still works for float-like.
        self.iters.append(it)

        for k, v in loss_dict.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(float(v))

        if it % self.freq != 0:
            return
        self._save_plot(it)

    def _save_plot(self, it: int):
        plt.figure()
        for k, ys in self.history.items():
            plt.plot(self.iters, ys, label=k)

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale("log")  # common for PINNs
        plt.title(f"Loss Curve (iter={it})")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(self.save_dir, "loss_curve.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
