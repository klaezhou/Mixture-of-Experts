# viz/callbacks.py
from __future__ import annotations
from typing import Dict, Any

class Callback:
    # The input type of the subclass must be consistent with that of the superclass
    """        
    self.args = args
    self.save_dir = args.save_dir
    self.freq_dict = freq_dict
    self.freq = None
    self.keep_last = keep_last
    self.iters=[]
    """
    def __init__(self):
        self._base_freq = 1000  # default frequency if not specified
    def on_train_begin(self, trainer): ...
    def on_iter_end(self, trainer, it: int, loss_dict: Dict[str, float]): ...
    def on_train_end(self, trainer): ...
    
    def on_phase_begin(self, trainer, phase_name: str): ...
