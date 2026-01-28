# viz/checkpoint_callback.py
import os
import torch
from viz.callbacks import Callback

class CheckpointCallback(Callback):
    def __init__(self, args,  freq=10000, keep_last=True):
        self.args = args
        self.save_dir = args.save_dir
        self.freq = freq
        self.keep_last = keep_last

        os.makedirs(self.save_dir, exist_ok=True)

    def on_iter_end(self, trainer, it: int, loss_dict: dict):
        if it % self.freq != 0:
            return

        log_dir = os.path.join(self.save_dir, "_log_model")
        os.makedirs(log_dir, exist_ok=True)

        

        payload = {
            "iter": it,
            "model": trainer.model.state_dict(),
        }
        torch.save(
            payload,
            os.path.join(log_dir, f"ckpt_iter_{it:06d}.pt")
        )


        # optional: also maintain a "last.pt" that always points to latest
        if self.keep_last:
            last_path = os.path.join(self.save_dir, "last.pt")
            torch.save(payload, last_path)
