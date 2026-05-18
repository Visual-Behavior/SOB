import torch
import torch.nn as nn
from torch import Tensor


def static_flow_inv(mu, sigma, T, K_target, K_source):
    """Compute optical flow for static scene with inverse depth.

    This function projects points from a target camera to a source camera using inverse depth
    representation and camera intrinsics/extrinsics.

    Args:
        mu (torch.Tensor): Inverse depth (disparity) means, shape [B, 1, H, W].
        sigma (torch.Tensor, optional): Inverse depth uncertainties, shape [B, 1, H, W].
            If not None, uncertainties will be propagated.
        T (torch.Tensor): Transformation matrices from target to source, shape [B, 4, 4].
        K_target (torch.Tensor): Intrinsic camera matrices of target views, shape [B, 3, 3].
        K_source (torch.Tensor): Intrinsic camera matrices of source views, shape [B, 3, 3].

    Returns:
        torch.Tensor or tuple: If sigma is None, returns the projected coordinates of shape [B, 2, H, W].
            Otherwise, returns a tuple of (projected coordinates, projected uncertainties) both of shape [B, 2, H, W].
    """
    B, _, H, W = mu.shape
    K_inv = torch.linalg.inv(K_target)

    x_coords = torch.arange(W, dtype=torch.float32, device=mu.device).view(1, W).expand(H, W)
    y_coords = torch.arange(H, dtype=torch.float32, device=mu.device).view(H, 1).expand(H, W)
    ones = torch.ones(H, W, dtype=torch.float32, device=mu.device)
    grid = torch.stack([x_coords, y_coords, ones], 0).view(1, 3, -1).expand(B, -1, -1)  # B 3 H*W
    R, t = T[:, :3, :3], T[:, :3, 3:4]

    mu = mu.view(B, 1, -1)

    # Standard projection of the mean to get the position
    mu_p = (K_source @ R @ K_inv) @ grid + K_source @ (mu * t)
    mu_p = mu_p[:, :2] / (mu_p[:, 2:] + 1e-7)

    if sigma is not None:
        # Approximation of position's variance
        R, t = R.detach(), t.detach()
        rot = (R @ K_inv) @ grid
        ri, ti = K_source @ rot, K_source @ t
        num = ti[:, :2] * ri[:, 2:] - ti[:, 2:] * ri[:, :2]
        denum = torch.square(K_source[:, 2:] @ (rot + mu.detach() * t)) + 1e-7
        sigma_p = sigma.view(B, 1, -1) * (num / denum).detach()

        return mu_p.view(B, 2, H, W), sigma_p.view(B, 2, H, W)

    return mu_p.view(B, 2, H, W)


def static_flow(mu, sigma, T, K_target, K_source):
    """Compute optical flow for static scene with depth.

    This function projects points from a target camera to a source camera using direct depth
    representation and camera intrinsics/extrinsics.

    Args:
        mu (torch.Tensor): Depth means, shape [B, 1, H, W].
        sigma (torch.Tensor, optional): Depth uncertainties, shape [B, 1, H, W].
            If not None, uncertainties will be propagated.
        T (torch.Tensor): Transformation matrices from target to source, shape [B, 4, 4].
        K_target (torch.Tensor): Intrinsic camera matrices of target views, shape [B, 3, 3].
        K_source (torch.Tensor): Intrinsic camera matrices of source views, shape [B, 3, 3].

    Returns:
        torch.Tensor or tuple: If sigma is None, returns the projected coordinates of shape [B, 2, H, W].
            Otherwise, returns a tuple of (projected coordinates, projected uncertainties) both of shape [B, 2, H, W].
    """
    B, _, H, W = mu.shape
    K_inv = torch.linalg.inv(K_target)
    R, t = T[:, :3, :3], T[:, :3, 3:4]

    x_coords = torch.arange(W, dtype=torch.float32, device=mu.device).view(1, W).expand(H, W)
    y_coords = torch.arange(H, dtype=torch.float32, device=mu.device).view(H, 1).expand(H, W)
    ones = torch.ones(H, W, dtype=torch.float32, device=mu.device)
    grid = torch.stack([x_coords, y_coords, ones], 0).view(1, 3, -1).expand(B, -1, -1)  # B 3 H*W

    mu = mu.view(B, 1, -1)

    kx, ti = (K_source @ R @ K_inv) @ grid, K_source @ t
    y = mu * kx + ti

    # Standard projection of the mean to get the position
    mu_p = y[:, :2] / (y[:, 2:] + 1e-7)

    if sigma is not None:
        # Approximation of position's variance
        y = y.detach()
        ti = ti.detach()
        num = y[:, 2:] * ti[:, :2] - ti[:, 2:] * y[:, :2]
        denum = torch.square(y[:, 2:]) + 1e-7
        sigma_p = sigma.view(B, 1, -1) * (num / denum).detach()

        return mu_p.view(B, 2, H, W), sigma_p.view(B, 2, H, W)

    return mu_p.view(B, 2, H, W)


class BackprojectDepth(nn.Module):
    """Module to backproject a depth map into a pointcloud.

    :param shape: (tuple[int, int]) Depth map shape as (height, width).
    """

    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.h, self.w = shape
        self.ones = nn.Parameter(torch.ones(1, 1, self.h * self.w), requires_grad=False)

        grid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing="xy")  # (h, w), (h, w)
        pix = torch.stack(grid).view(2, -1)[None]  # (1, 2, h*w) as (x, y)
        pix = torch.cat((pix, self.ones), dim=1)  # (1, 3, h*w)
        self.pix = nn.Parameter(pix, requires_grad=False)

    def forward(self, depth: Tensor, K_inv: Tensor) -> Tensor:
        """Backproject a depth map into a pointcloud.

        Camera is assumed to be at the origin.

        :param depth: (Tensor) (b, 1, h, w) Depth map to backproject.
        :param K_inv: (Tensor) (b, 4, 4) Inverse camera intrinsic parameters.
        :return: (Tensor) (b, 4, h*w) Backprojected 3-D points as (x, y, z, homo).
        """
        b = depth.shape[0]
        pts = K_inv[:, :3, :3] @ self.pix.repeat(b, 1, 1)  # (b, 3, h*w) Cam rays.
        pts *= depth.flatten(-2)  # 3D points.
        pts = torch.cat((pts, self.ones.repeat(b, 1, 1)), dim=1)  # (b, 4, h*w) Add homogenous.
        return pts
