# viz/error_plot_callback.py
import os
from viz.callbacks import Callback

class ErrorPlotCallback(Callback):
    def __init__(self, args, equation, freq=500):
        self.args = args
        self.eq = equation
        self.save_dir = args.save_dir
        self.freq = freq

        os.makedirs(self.save_dir, exist_ok=True)

    def on_iter_end(self, trainer, it: int, loss_dict: dict):
        if it % self.freq != 0:
            return
        self.eq.plot_error(trainer.model, it, self.save_dir)
        
    def on_train_begin(self, trainer):
        self.eq.plot_ground_truth( self.save_dir)
        
    def on_train_end(self, trainer):
        self.eq.plot_error(trainer.model, -1, self.save_dir)
        self.eq.plot_u(trainer.model, self.save_dir)