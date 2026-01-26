import torch.nn as nn
from abc import ABC, abstractmethod
# Equation/_base.py
class BaseEquation:
    def __init__(self, args):
        self.args = args
    @abstractmethod
    def compute_loss(self, model, batch: dict):
        """
        batch is a dict containing necessary data, e.g.,
        batch["X_f"], batch["X_bnd"], batch["y_bnd"] ...
        return a dict of losses, e.g.,
        {
            "total": total_loss, with grad
            "pde": pde_loss, item(),
            "bc": boundary_loss, item()
        }
        """
        pass
    
    @abstractmethod
    def get_data(self, data_loader):
        """
        return dict(batch) including necessary data for compute_loss
        e.g., {"X_f":..., "X_bnd":..., "y_bnd":...}
        """
        pass

    
    