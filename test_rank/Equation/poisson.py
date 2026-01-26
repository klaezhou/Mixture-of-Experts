# Equation/poisson.py
import torch
from Equation._base import BaseEquation
import math

class PoissonEquation(BaseEquation):
    """
    Poisson equation -Δu = f, on domain Ω , with Dirichler boundary condtion u=g at ∂Ω. 
    two dimensional case.
    """
    def f(self, x):
        """
        right term f(x)
        x: [N, dim]
        return: [N, 1]
        """
        # example: f = 2*pi^2*sin(pi*x1)*sin(pi*x2)

        x1 = x[:, 0:1]
        x2 = x[:, 1:2]
        return 2 * (math.pi**2) * torch.sin(math.pi * x1) * torch.sin(math.pi * x2)

    def g(self, x):
        """
        boundary value g(x). u=g at boundary
        x: [N, dim]
        return: [N, 1]
        """
        # example: u=0 at boundary
        return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
    def laplacian(self, u, x):
        """
        u: [N,1] = model(x)
        x: [N,dim] requires_grad=True
        return: [N,1]  Laplacian(u)
        """
        grad_u = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]  # [N, dim]

        lap = 0.0
        for d in range(x.shape[1]):
            grad_ud = grad_u[:, d:d+1]  # [N,1]
            grad2_ud = torch.autograd.grad(
                outputs=grad_ud, inputs=x,
                grad_outputs=torch.ones_like(grad_ud),
                create_graph=True, retain_graph=True
            )[0][:, d:d+1]  # [N,1]
            lap = lap + grad2_ud
        return lap

    def compute_loss(self, model, batch: dict):
        
        # 1) PDE residual
        x_f = batch["X_f"].requires_grad_(True)  
        u_f = model(x_f)
        lap_u = self.laplacian(u_f, x_f)
        f_f = batch.get("f_f", self.f(x_f))

        r_f = (-lap_u - f_f)
        loss_pde = torch.mean(r_f**2)

        # 2) Boundary loss
        x_b = batch["X_b"]
        u_b = model(x_b)
        g_b = batch.get("g_b", self.g(x_b))
        loss_bc = torch.mean((u_b - g_b)**2)

        # 3) Total and Weighting
        w_pde = getattr(self.args, "w_pde", 1.0)
        w_bc  = getattr(self.args, "w_bc", 1.0)
        
        total_loss = w_pde * loss_pde + w_bc * loss_bc

        # --- 关键修改：返回字典 ---
        loss_dict = {
        "total": total_loss,
        "pde": loss_pde.detach(),
        "bc": loss_bc.detach(),
        }
        return loss_dict
    
    def get_data(self, data_loader):
        # 1) decide how many points to sample
        Nf = getattr(self.args, "Nf", 10000)
        Nb = getattr(self.args, "Nb", 2000)

        # 2) sample interior/boundary points using DataLoader tools
        sample_method = getattr(self.args, "sample_method", "grid") # "random" or "grid"
        #square domain [-1,1]x[-1,1] 
        if sample_method == "random":
            X_f = data_loader.sample_interior_box(Nf, dim=2, low=-1.0, high=1.0)
            X_b = data_loader.sample_boundary_box_2d(Nb, low=-1.0, high=1.0)
        elif sample_method == "grid":
            nx = getattr(self.args, "nx", 100)
            ny = getattr(self.args, "ny", 100)
            n_per_edge = getattr(self.args, "n_per_edge", 100)
            X_f = data_loader.sample_interior_grid_2d(nx=nx, ny=ny, low=-1.0, high=1.0, exclude_boundary=True)
            X_b = data_loader.sample_boundary_grid_2d(n_per_edge=n_per_edge, low=-1.0, high=1.0, include_corners=True)
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        # 3) (recommended) precompute f and g values
        f_f = self.f(X_f)
        g_b = self.g(X_b)

        # 4) return a unified batch dict
        return {"X_f": X_f, "X_b": X_b, "f_f": f_f, "g_b": g_b}
    
    def exact_solution(self, x):
            """return exact solution at x for error analysis"""
            
            return torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])

    def get_test_data(self):
        """generate test data for evaluation"""
        pass