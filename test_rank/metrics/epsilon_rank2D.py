# metrics/epsilon_rank2D.py
# 2D Epsilon-Rank computation with trapezoidal quadrature
"""def epsilon_rank_model_2d_trapz(
    model,
    feature_getter,
    nx: int,
    ny: int,
    low: float,
    high: float,
    eps: float = 1e-8,
)"""
import torch


def make_grid_2d(nx: int, ny: int, lowx: float, highx: float, lowy: float, highy: float, device=None, dtype=None):
    """
    Create a 2D tensor grid and flatten it.

    Returns:
        x_grid: [N, 2] with N = nx * ny
        X, Y:   [nx, ny] meshgrid (CPU/GPU consistent)
    """
    xs = torch.linspace(lowx, highx, steps=nx, device=device, dtype=dtype)
    ys = torch.linspace(lowy, highy, steps=ny, device=device, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")   # [nx, ny]
    x_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N,2]

    return x_grid, X, Y


def trapezoid_weights_2d(nx: int, ny: int, lowx: float, highx: float, lowy: float, highy: float, device=None, dtype=None):
    """
    2D trapezoidal rule weights on [lowx, highx] x [lowy, highy].

    Returns:
        w:     [N] weights
        area:  scalar = sum(w) = (highx-lowx) * (highy-lowy)
    """
    dx = (highx - lowx) / (nx - 1)
    dy = (highy - lowy) / (ny - 1)
    wx = torch.ones(nx, device=device, dtype=dtype)
    wy = torch.ones(ny, device=device, dtype=dtype)
    wx[0] *= 0.5
    wx[-1] *= 0.5
    wy[0] *= 0.5
    wy[-1] *= 0.5

    W = torch.outer(wx, wy) * (dx * dy)   # [nx, ny]
    w = W.reshape(-1)                     # [N]
    area = W.sum()
    return w, area


@torch.no_grad()
def epsilon_rank_model_2d_trapz_auto(
    model,
    feature_getter,
    nx: int, ny: int,
    lowx: float, highx: float,
    lowy: float, highy: float,
    eps: float = 1e-3,
):
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    # 1) grid
    x_grid, _, _ = make_grid_2d(nx, ny, lowx, highx, lowy, highy, device=device, dtype=dtype)

    # 2) feature getter
    out = feature_getter(model, x_grid)

    if isinstance(out, (tuple, list)) and len(out) == 2:
        Phi, weight = out
    else:
        Phi, weight = out, None

    if Phi.dim() == 1:
        Phi = Phi.unsqueeze(1)

    N, m = Phi.shape
    if N != nx * ny:
        raise RuntimeError(f"Feature shape mismatch: N={N}, nx*ny={nx*ny}")

    # 3) trapz weights
    w, area = trapezoid_weights_2d(nx, ny, lowx, highx, lowy, highy, device=device, dtype=dtype)
    w = w.reshape(-1, 1)
    area = torch.as_tensor(area, device=device, dtype=dtype)

    # 4) build Gram
    if weight is None:
        Phi_w = Phi * w                          # [N,m]
    else:
        # make weight shape compatible with Phi
        if weight.dim() == 1:
            weight = weight.reshape(-1, 1)        # [N,1]
        elif weight.dim() == 2 and weight.shape[1] == 1:
            pass                                 # [N,1]
        # allow [N,m] directly
        if weight.shape[0] != N:
            raise RuntimeError(f"weight N mismatch: weight.shape={weight.shape}, Phi.shape={Phi.shape}")

        # If weight is [N,1], it broadcasts to [N,m]. If weight is [N,m], elementwise works.
        Phi_w = Phi * (weight * w)               # [N,m]

    G = (Phi.T @ Phi_w) / (area + 1e-12)
    evals = torch.linalg.eigvalsh(G)
    rank = int((evals > eps).sum().item())
    return rank, evals

    

