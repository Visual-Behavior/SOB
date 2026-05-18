import torch
from torch import nn


class SSIM(nn.Module):
    """SSIM (Structural Similarity Index) loss with uncertainty propagation.

    This module computes the SSIM loss between two images and can optionally
    propagate uncertainty from input to output.

    Attributes:
        pool (nn.AvgPool2d): Pooling layer for local statistics.
        refl (nn.ReflectionPad2d): Reflection padding for border handling.
        C1 (float): Constant for numerical stability in luminance comparison.
        C2 (float): Constant for numerical stability in contrast comparison.
    """

    def __init__(self):
        """Initialize the SSIM loss module."""
        super().__init__()
        self.pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, y_mu, x_mu, x_sigma=None):
        """Compute SSIM loss and optionally propagate uncertainty.

        Args:
            y_mu (torch.Tensor): First image, shape [B, C, H, W].
            x_mu (torch.Tensor): Second image, shape [B, C, H, W].
            x_sigma (torch.Tensor, optional): Uncertainty of second image,
                shape [B, C, H, W]. Defaults to None.

        Returns:
            tuple or torch.Tensor: If x_sigma is not None, returns
                (ssim_loss, ssim_sigma) where ssim_loss is the SSIM loss and
                ssim_sigma is the propagated uncertainty. Otherwise, returns
                just the ssim_loss.
        """
        x = self.refl(x_mu)
        y = self.refl(y_mu)

        # Means
        mu_x = self.pool(x)
        mu_y = self.pool(y)

        # Variances and covariance
        sigma_x2 = self.pool(x**2) - mu_x**2
        sigma_y2 = self.pool(y**2) - mu_y**2
        sigma_xy = self.pool(x * y) - mu_x * mu_y

        #           (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)       A * B   P
        # SSIM = ––––––––––––––––––––––––––––––––––––––––––––––––––– = ––––– = –
        #        (mu_x^2 + mu_y^2 + C1) * (sigma_x2 + sigma_y2 + C2)   C * D   Q

        # SSIM components
        A = 2 * mu_x * mu_y + self.C1
        B = 2 * sigma_xy + self.C2
        C = mu_x**2 + mu_y**2 + self.C1
        D = sigma_x2 + sigma_y2 + self.C2

        # SSIM calculation
        P = A * B
        Q = C * D
        ssim = P / Q
        ssim_loss = torch.clamp((1 - ssim) / 2, 0, 1)

        if x_sigma is not None:
            # Standard deviation calculation
            N = 9  # 3x3 window

            # Computing partial derivatives
            d1 = mu_y * B * Q
            d2 = mu_x * D * P
            d3 = A * Q * (y_mu - mu_y)
            d4 = C * P * (x_mu - mu_x)

            # Final standard deviation
            ssim_sigma = x_sigma * (torch.abs(d1 - d2 + d3 - d4) / (N * Q**2)).detach()

            return ssim_loss, ssim_sigma
        return ssim_loss
