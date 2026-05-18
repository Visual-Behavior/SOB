from kornia.filters import laplacian
from torch import nn
import torch
import torch.nn.functional as F


class SmoothnessLoss(nn.Module):
    """Edge-aware smoothness loss for depth estimation.

    This loss penalizes depth discontinuities.

    Attributes:
        weight (float): Weight factor for the smoothness loss.
    """

    def __init__(self, mu_weight, sigma_weight=0):
        """Initialize the smoothness loss.

        Args:
            weight (float): Weight factor for the smoothness loss.
        """
        super().__init__()
        self.mu_w = mu_weight
        self.sigma_w = sigma_weight
        self.mu = mu_weight > 0
        self.sigma = sigma_weight > 0

    def weighted_laplacian(self, mu, alpha):
        """Compute weighted Laplacian for depth smoothness.

        Args:
            mu (torch.Tensor): Depth mean values, shape [B, C, H, W].
            alpha (torch.Tensor): Alpha weights, shape [B, C, H, W].

        Returns:
            torch.Tensor: Weighted Laplacian values, shape [B, H, W].
        """
        lap = laplacian(mu, 3).abs() / mu.detach().mean()
        return (lap * alpha.detach()).sum(dim=1)

    def forward(self, multi_scale_depth, target):
        """Compute depth smoothness loss across multiple scales.

        Applies the weighted Laplacian to depth at multiple scales,
        with larger scales having lower weight.

        Args:
            multi_scale_depth (list): List of depth outputs at different scales.
                Each element is a dictionary with keys 'mu', 'sigma', 'alpha'.

        Returns:
            torch.Tensor: Weighted smoothness loss value.
        """
        loss = torch.tensor(0.0, device=target.device)
        for scale, mixture in enumerate(multi_scale_depth):

            mu, sigma = mixture["mu"], mixture["sigma"]
            # loss += self.weighted_laplacian(mu, alpha).mean() / 2**scale
            if self.mu:
                mu = F.interpolate(mu, target.shape[-2:], mode="bilinear", align_corners=False)
                loss += self.mu_w * get_smooth_loss(mu, target).mean() / 2**scale

            if self.sigma:
                sigma = F.interpolate(torch.log(sigma), target.shape[-2:], mode="bilinear", align_corners=False)
                loss += self.sigma_w * get_smooth_loss(sigma, target).mean() / 2**scale

        return loss


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """

    mean_disp = disp.mean(dim=(2, 3), keepdim=True)
    disp = disp / (mean_disp + 1e-7)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()
