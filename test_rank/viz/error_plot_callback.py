# viz/error_plot_callback.py
import os
from model.moe_d import MOE_dense
from viz.callbacks import Callback

class ErrorPlotCallback(Callback):
    def __init__(self, args, equation, freq_dict={"adam":500,"lbfgs":50}):
        super().__init__()
        self.args = args
        self.eq = equation
        self.save_dir = args.save_dir
        self.freq_dict = freq_dict
        self.freq = None
        os.makedirs(self.save_dir, exist_ok=True)

    def on_iter_end(self, trainer, it: int, loss_dict: dict):
        if it % self.freq != 0:
            return
        self.eq.plot_error(trainer.model, it, self.save_dir)
        if isinstance(trainer.model, MOE_dense):
            self.eq.plot_gate(trainer.model, it, self.save_dir)

        
    def on_train_begin(self, trainer):
        self.eq.plot_ground_truth( self.save_dir)
        
    def on_train_end(self, trainer):
        self.eq.plot_error(trainer.model, -1, self.save_dir)
        self.eq.plot_u(trainer.model, self.save_dir)
        
    def on_phase_begin(self, trainer, phase: str):
        if phase not in ("adam", "lbfgs"):
            print(f"[Warning] Unknown phase '{phase}', using base freq.")
        self.freq = self.freq_dict.get(phase, self._base_freq)