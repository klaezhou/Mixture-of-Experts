# data/data_loader.py
import torch

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.device = args.device

    def sample_interior_box(self, n: int, dim: int = 2, low=0.0, high=1.0):
        """Uniformly sample points inside the (low, high)^dim box."""
        x = low + (high - low) * torch.rand(n, dim, device=self.device)
        return x

    def sample_boundary_box_2d(self, n: int, low=0.0, high=1.0):
        """
        Randomly sample n points on the boundary of a 2D square/box.
        Returns a tensor of shape [n, 2].
        """
        t = low + (high - low) * torch.rand(n, 1, device=self.device)

        # Four edges: x=low, x=high, y=low, y=high
        X1 = torch.cat([torch.full_like(t, low),  t], dim=1)
        X2 = torch.cat([torch.full_like(t, high), t], dim=1)
        X3 = torch.cat([t, torch.full_like(t, low)], dim=1)
        X4 = torch.cat([t, torch.full_like(t, high)], dim=1)

        X = torch.cat([X1, X2, X3, X4], dim=0)

        # Truncate to n points if we created more than needed
        return X[:n]

    def sample_interior_grid_2d(self, nx: int, ny: int, low=0.0, high=1.0, exclude_boundary: bool = True):
        """
        Equidistant grid sampling in a 2D box. Returns points of shape [N, 2].
        If exclude_boundary=True, boundary points are excluded (interior only).
        """
        if exclude_boundary:
            xs = torch.linspace(low, high, steps=nx + 2, device=self.device)[1:-1]
            ys = torch.linspace(low, high, steps=ny + 2, device=self.device)[1:-1]
        else:
            xs = torch.linspace(low, high, steps=nx, device=self.device)
            ys = torch.linspace(low, high, steps=ny, device=self.device)

        X, Y = torch.meshgrid(xs, ys, indexing="ij")  # [nx, ny]
        pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # [N, 2]
        return pts

    def sample_boundary_grid_2d(self, n_per_edge: int, low=0.0, high=1.0, include_corners: bool = True):
        """
        Equidistant sampling on the boundary of a 2D square/box.
        Returns points of shape [N, 2], where N is approximately 4*n_per_edge.
        Corner duplication and deduplication are controlled by include_corners.
        """
        if include_corners:
            t = torch.linspace(low, high, steps=n_per_edge, device=self.device).unsqueeze(1)  # [n, 1]
        else:
            # Exclude corners by removing both endpoints
            t = torch.linspace(low, high, steps=n_per_edge + 2, device=self.device)[1:-1].unsqueeze(1)

        left   = torch.cat([torch.full_like(t, low),  t], dim=1)   # x = low
        right  = torch.cat([torch.full_like(t, high), t], dim=1)   # x = high
        bottom = torch.cat([t, torch.full_like(t, low)], dim=1)    # y = low
        top    = torch.cat([t, torch.full_like(t, high)], dim=1)   # y = high

        pts = torch.cat([left, right, bottom, top], dim=0)

        # If include_corners=True, corner points can appear multiple times
        # (e.g., (low, low) appears on both the left and bottom edges).
        # Optionally deduplicate:
        if include_corners:
            pts = torch.unique(pts, dim=0)

        return pts
