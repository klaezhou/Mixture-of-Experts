# viz/checkpoint_callback.py
import os
import torch
from viz.callbacks import Callback

class CheckpointCallback(Callback):
    def __init__(self, args,  freq_dict={"adam":10000,"lbfgs":200}, keep_last=True):
        super().__init__()
        self.args = args
        self.save_dir = args.save_dir
        self.freq_dict = freq_dict
        self.freq = None
        self.keep_last = keep_last
        self.iters=[]
        os.makedirs(self.save_dir, exist_ok=True)

    def on_iter_end(self, trainer, it: int, loss_dict: dict):
        self.iters.append(it)
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
            last_path = os.path.join(log_dir, "last.pt")
            torch.save(payload, last_path)

    def on_train_end(self, trainer):
        # save one final checkpoint at the end of training
        it = self.iters[-1]
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

        if self.keep_last:
            last_path = os.path.join(log_dir, "last.pt")
            torch.save(payload, last_path)
            
    def on_phase_begin(self, trainer, phase: str):
        if phase not in ("adam", "lbfgs"):
            print(f"[Warning] Unknown phase '{phase}', using base freq.")
        self.freq = self.freq_dict.get(phase, self._base_freq)