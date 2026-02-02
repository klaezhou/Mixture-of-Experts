# Equation/allen_cahn.py
import os
import math
import torch
import matplotlib.pyplot as plt
from Equation._base import BaseEquation


class AllenCahnEquation(BaseEquation):
    """
    1D Allen–Cahn with manufactured source term (so we have an exact u(t,x)):

        u_t - eps^2 * u_xx + u^3 - u = s(t,x),
        t in (0, 1), x in (-1, 1),
        periodic BC: u(t,-1)=u(t,1) and u_x(t,-1)=u_x(t,1),
        IC: u(0,x)=u_exact(0,x).

    Input x is [N, 2] with columns: [t, x].
    """

    # ---------- exact solution config ----------
    def _get_eps(self):
        return getattr(self.args, "eps", 0.01)

    def _get_mode_n(self):
        return getattr(self.args, "mode_n", 1)

    def _get_alpha(self):
        return getattr(self.args, "alpha", 1.0)

    # ---------- exact solution ----------
    def exact_solution(self, X):
        """
        X: [N,2] where X[:,0]=t, X[:,1]=x
        return: [N,1]
        """
        t = X[:, 0:1]
        x = X[:, 1:2]
        n = self._get_mode_n()
        alpha = self._get_alpha()
        return torch.exp(-alpha * t) * torch.cos(n * math.pi * x)

    # ---------- source term s(t,x) for manufactured solution ----------
    def source(self, X):
        """
        s = u_t - eps^2 u_xx + u^3 - u, with u = exact_solution
        so that u_exact is the exact solution of the forced Allen-Cahn.
        """
        eps = self._get_eps()
        n = self._get_mode_n()
        alpha = self._get_alpha()

        u = self.exact_solution(X)
        t = X[:, 0:1]
        x = X[:, 1:2]

        # u_t = -alpha * u
        u_t = -alpha * u

        # u_xx = -(n*pi)^2 * u
        u_xx = -(n * math.pi) ** 2 * u

        # s = u_t - eps^2 u_xx + u^3 - u
        s = u_t - (eps ** 2) * u_xx + (u ** 3) - u
        return s

    # ---------- derivatives ----------
    def u_t(self, u, X):
        """time derivative ∂u/∂t"""
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]  # [N,2]
        return grad_u[:, 0:1]

    def u_x(self, u, X):
        """space derivative ∂u/∂x"""
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        return grad_u[:, 1:2]

    def u_xx(self, u, X):
        """space second derivative ∂²u/∂x²"""
        ux = self.u_x(u, X)
        grad_ux = torch.autograd.grad(
            outputs=ux,
            inputs=X,
            grad_outputs=torch.ones_like(ux),
            create_graph=True,
            retain_graph=True
        )[0]
        return grad_ux[:, 1:2]

    # ---------- loss ----------
    def compute_loss(self, model, batch: dict):

        eps = self._get_eps()

        # 1) PDE residual on interior points
        X_f = batch["X_f"].requires_grad_(True)  # [Nf,2]
        u_f = model(X_f)
        if u_f.dim() == 1:
            u_f = u_f.unsqueeze(1)

        ut = self.u_t(u_f, X_f)
        uxx = self.u_xx(u_f, X_f)

        s_f = batch.get("s_f", self.source(X_f))
        r_f = ut - (eps ** 2) * uxx + (u_f ** 3) - u_f - s_f
        loss_pde = torch.mean(r_f ** 2)

        # 2) Periodic BC: u(t,-1)=u(t,1) and u_x(t,-1)=u_x(t,1)
        X_bL = batch["X_bL"].requires_grad_(True)  # [Nb,2]  x=-1
        X_bR = batch["X_bR"].requires_grad_(True)  # [Nb,2]  x=+1

        u_bL = model(X_bL)
        u_bR = model(X_bR)
        if u_bL.dim() == 1:
            u_bL = u_bL.unsqueeze(1)
        if u_bR.dim() == 1:
            u_bR = u_bR.unsqueeze(1)

        ux_bL = self.u_x(u_bL, X_bL)
        ux_bR = self.u_x(u_bR, X_bR)

        loss_bc_u = torch.mean((u_bL - u_bR) ** 2)
        loss_bc_ux = torch.mean((ux_bL - ux_bR) ** 2)
        loss_bc = loss_bc_u + loss_bc_ux

        # 3) Initial condition u(0,x)=u_exact(0,x)
        X_i = batch["X_i"]  # [Ni,2], t=0
        u_i_pred = model(X_i)
        if u_i_pred.dim() == 1:
            u_i_pred = u_i_pred.unsqueeze(1)
        u_i_true = batch.get("u_i", self.exact_solution(X_i))
        loss_ic = torch.mean((u_i_pred - u_i_true) ** 2)

        # 4) Total
        w_pde = getattr(self.args, "w_pde", 1.0)
        w_bc  = getattr(self.args, "w_bc", 1.0)
        w_ic  = getattr(self.args, "w_ic", 1.0)

        total_loss = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic

        return {
            "total": total_loss,
            "pde": loss_pde.detach(),
            "bc": loss_bc.detach(),
            "ic": loss_ic.detach(),
        }

    # ---------- data ----------
    def get_data(self, data_loader=None):
        """
        返回 batch:
          X_f: interior collocation points in (t,x)
          X_bL, X_bR: periodic boundary pairs at x=-1 and x=+1
          X_i: initial points at t=0
          s_f: precomputed source on X_f
          u_i: precomputed initial true value on X_i
        """

        device = self.args.device

        # sizes
        Nf = getattr(self.args, "Nf", 10000)
        Nb = getattr(self.args, "Nb", 2000)
        Ni = getattr(self.args, "Ni", 1000)

        sample_method = getattr(self.args, "sample_method", "grid")  # random or grid

        t0, t1 = 0.0, 1.0
        x0, x1 = -1.0, 1.0

        # ---- interior points ----
        if sample_method == "random":
            t_f = t0 + (t1 - t0) * torch.rand((Nf, 1), device=device)
            x_f = x0 + (x1 - x0) * torch.rand((Nf, 1), device=device)
            X_f = torch.cat([t_f, x_f], dim=1)
        elif sample_method == "grid":
            nt = getattr(self.args, "nt", 100)
            nx = getattr(self.args, "nx", 200)
            ts = torch.linspace(t0, t1, steps=nt, device=device)
            xs = torch.linspace(x0, x1, steps=nx, device=device)
            T, X = torch.meshgrid(ts, xs, indexing="ij")
            X_f = torch.stack([T.reshape(-1), X.reshape(-1)], dim=1)
        else:
            raise ValueError(f"Unknown sample_method: {sample_method}")

        # ---- periodic boundary pairs ----
        # sample times, set x=-1 and x=1
        t_b = t0 + (t1 - t0) * torch.linspace(0.0, 1.0, steps=Nb, device=device).reshape(-1, 1)
        X_bL = torch.cat([t_b, torch.full_like(t_b, x0)], dim=1)
        X_bR = torch.cat([t_b, torch.full_like(t_b, x1)], dim=1)

        # ---- initial points ----
        x_i = x0 + (x1 - x0) * torch.linspace(0.0, 1.0, steps=Ni, device=device).reshape(-1, 1)
        t_i = torch.full_like(x_i, t0)
        X_i = torch.cat([t_i, x_i], dim=1)


        # precompute
        s_f = self.source(X_f)
        u_i = self.exact_solution(X_i)

        return {
            "X_f": X_f,
            "X_bL": X_bL,
            "X_bR": X_bR,
            "X_i": X_i,
            "s_f": s_f,
            "u_i": u_i,
        }

    # ---------- plotting / error ----------
    @torch.no_grad()
    def plot_error(self, model, it: int, save_dir: str):
        """
        Plot |u_pred - u_exact| on a (t,x) grid as a heatmap,
        and log L2 errors.
        """
        os.makedirs(save_dir, exist_ok=True)
        device = self.args.device

        nt = getattr(self.args, "eval_grid_nt", 120)
        nx = getattr(self.args, "eval_grid_nx", 240)

        t0, t1 = 0.0, 1.0
        x0, x1 = -1.0, 1.0
        eps = 1e-12

        ts = torch.linspace(t0, t1, steps=nt, device=device)
        xs = torch.linspace(x0, x1, steps=nx, device=device)
        T, X = torch.meshgrid(ts, xs, indexing="ij")  # [nt,nx]
        grid_tx = torch.stack([T.reshape(-1), X.reshape(-1)], dim=1)  # [N,2]

        model_was_training = model.training
        model.eval()

        pred = model(grid_tx)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)

        exact = self.exact_solution(grid_tx)
        err = pred - exact

        l2_abs = torch.sqrt(torch.mean(err ** 2)).item()
        denom = torch.sqrt(torch.mean(exact ** 2)).item()
        l2_rel = l2_abs / (denom + eps)

        err_abs_grid = err.abs().reshape(nt, nx).detach().cpu()

        # plot
        plt.figure(figsize=(7, 4))
        plt.title(f"|u_pred - u_exact| (iter={it})\nL2_abs={l2_abs:.3e}, L2_rel={l2_rel:.3e}")
        plt.pcolormesh(xs.detach().cpu(), ts.detach().cpu(), err_abs_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.tight_layout()

        img_path = os.path.join(save_dir, f"allen_cahn_error_iter_{it:06d}.png")
        plt.savefig(img_path, dpi=150)
        plt.close()

        # log
        csv_path = os.path.join(save_dir, "allen_cahn_error_log.csv")
        need_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write("iter,l2_abs,l2_rel\n")
            f.write(f"{it},{l2_abs:.12e},{l2_rel:.12e}\n")

        print(f"[ErrorPlot] iter={it} | L2_abs={l2_abs:.3e} | L2_rel={l2_rel:.3e} | saved: {img_path}")

        if model_was_training:
            model.train()

    @torch.no_grad()
    def plot_ground_truth(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        device = self.args.device

        nt = getattr(self.args, "eval_nt", 120)
        nx = getattr(self.args, "eval_nx", 240)

        t0, t1 = 0.0, 1.0
        x0, x1 = -1.0, 1.0

        ts = torch.linspace(t0, t1, steps=nt, device=device)
        xs = torch.linspace(x0, x1, steps=nx, device=device)
        T, X = torch.meshgrid(ts, xs, indexing="ij")
        grid_tx = torch.stack([T.reshape(-1), X.reshape(-1)], dim=1)

        exact = self.exact_solution(grid_tx).reshape(nt, nx).detach().cpu()

        plt.figure(figsize=(7, 4))
        plt.title("u_exact (ground truth)")
        plt.pcolormesh(xs.detach().cpu(), ts.detach().cpu(), exact, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "allen_cahn_ground_truth.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

    @torch.no_grad()
    def plot_u(self, model, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        device = self.args.device

        nt = getattr(self.args, "eval_nt", 120)
        nx = getattr(self.args, "eval_nx", 240)

        t0, t1 = 0.0, 1.0
        x0, x1 = -1.0, 1.0

        ts = torch.linspace(t0, t1, steps=nt, device=device)
        xs = torch.linspace(x0, x1, steps=nx, device=device)
        T, X = torch.meshgrid(ts, xs, indexing="ij")
        grid_tx = torch.stack([T.reshape(-1), X.reshape(-1)], dim=1)

        model_was_training = model.training
        model.eval()

        pred = model(grid_tx)
        if pred.dim() == 1:
            pred = pred.unsqueeze(1)
        pred_grid = pred.reshape(nt, nx).detach().cpu()

        plt.figure(figsize=(7, 4))
        plt.title("u_pred")
        plt.pcolormesh(xs.detach().cpu(), ts.detach().cpu(), pred_grid, shading="auto")
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("t")
        plt.tight_layout()

        out_path = os.path.join(save_dir, "allen_cahn_predict.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        if model_was_training:
            model.train()

    def plot_gate(self, model, it, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        device = self.args.device

        nt = getattr(self.args, "eval_nt", 120)
        nx = getattr(self.args, "eval_nx", 240)

        t0, t1 = 0.0, 1.0
        x0, x1 = -1.0, 1.0

        ts = torch.linspace(t0, t1, steps=nt, device=device)
        xs = torch.linspace(x0, x1, steps=nx, device=device)
        T, X = torch.meshgrid(ts, xs, indexing="ij")
        grid_tx = torch.stack([T.reshape(-1), X.reshape(-1)], dim=1)

        moe = model.model  # MOE_dense: self.model = self.moe
        with torch.no_grad():
            gates = moe.gating_network(grid_tx)     # [N, E]   

        E = gates.shape[1]

        # ---- plot ----
        fig, axes = plt.subplots(1, E, figsize=(4 * E, 4), squeeze=False)

        for e in range(E):
            gate_e = gates[:, e].reshape(nt, nx).cpu().numpy()
            ax = axes[0, e]
            im = ax.imshow(
                gate_e,
                origin="lower",
                extent=[x0, x1, t0, t1],
                cmap="viridis",
            )
            ax.set_title(f"Gate {e}")
            fig.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"MoE Gate Distribution (iter={it})")
        plt.tight_layout()

        # ---- save ----
        out_dir = save_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gate.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)