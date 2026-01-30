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


def make_grid_2d(nx: int, ny: int, low: float, high: float, device=None, dtype=None):
    """
    Create a 2D tensor grid and flatten it.

    Returns:
        x_grid: [N, 2] with N = nx * ny
        X, Y:   [nx, ny] meshgrid (CPU/GPU consistent)
    """
    xs = torch.linspace(low, high, steps=nx, device=device, dtype=dtype)
    ys = torch.linspace(low, high, steps=ny, device=device, dtype=dtype)

    X, Y = torch.meshgrid(xs, ys, indexing="ij")   # [nx, ny]
    x_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N,2]

    return x_grid, X, Y


def trapezoid_weights_2d(nx: int, ny: int, low: float, high: float, device=None, dtype=None):
    """
    2D trapezoidal rule weights on [low, high]^2.

    Returns:
        w:     [N] weights
        area:  scalar = sum(w) = (high-low)^2
    """
    dx = (high - low) / (nx - 1)
    dy = (high - low) / (ny - 1)

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
def epsilon_rank_model_2d_trapz(
    model,
    feature_getter,
    nx: int,
    ny: int,
    low: float,
    high: float,
    eps: float = 1e-3,
):
    """
    Full 2D epsilon-rank pipeline (grid + trapezoid rule).

    Steps:
      1) build 2D grid
      2) evaluate neuron features Phi
      3) build Gram matrix via trapezoidal quadrature
      4) compute epsilon-rank

    Returns:
        rank:  int
        evals: eigenvalues of Gram matrix
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # 1) grid
    x_grid, _, _ = make_grid_2d(nx, ny, low, high, device=device, dtype=dtype)

    # 2) neuron features
    Phi = feature_getter(model, x_grid)  # [N, m]
    # Phi = Phi - Phi.mean(dim=0, keepdim=True)  # zero-mean features

    N, m = Phi.shape
    if N != nx * ny:
        raise RuntimeError("Feature shape does not match grid size.")

    # 3) trapezoidal Gram matrix
    w, area = trapezoid_weights_2d(nx, ny, low, high, device=device, dtype=dtype)
    Phi_w = Phi * w.unsqueeze(1)          # [N, m]
    G = (Phi.T @ Phi_w) / area             # [m, m]

    # 4) eigenvalues & rank
    evals = torch.linalg.eigvalsh(G)
    rank = int((evals > eps).sum().item())

    return rank, evals



