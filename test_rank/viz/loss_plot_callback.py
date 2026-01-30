# viz/loss_plot_callback.py
import os
import matplotlib.pyplot as plt
from viz.callbacks import Callback

class LossPlotCallback(Callback):
    def __init__(self, args,  freq_dict={"adam":5,"lbfgs":1}):
        super().__init__()
        self.args = args
        self.save_dir =  args.save_dir
        self.freq_dict = freq_dict
        self.freq = None
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
        
    def on_train_end(self, trainer):
        self._save_plot(self.iters[-1])

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
        
    def on_phase_begin(self, trainer, phase: str):
        if phase not in ("adam", "lbfgs"):
            print(f"[Warning] Unknown phase '{phase}', using base freq.")
        self.freq = self.freq_dict.get(phase, self._base_freq)
