import torch
from torch import nn
from ..losses.ssim import SSIM
from ..distribution import gaussian
from kornia.filters import spatial_gradient


class MomentMatchingLoss(nn.Module):
    """Loss function for moment matching of depth distributions.

    This loss optimizes depth estimation by matching moments between
    the target color distribution and the predicted color distribution.

    Attributes:
        ssim (SSIM): Structural similarity index measure module.
        alpha_f: Alpha loss function.
        alpha_entropy (float): Weight for alpha entropy regularization.
        alpha_smooth (float): Weight for alpha smoothness regularization.
        sigma_loss (float): Weight for sigma loss.
    """

    def __init__(
        self,
        distribution="gaussian",
        alpha_f=None,
        alpha_entropy=0.0,
        alpha_smooth=0.0,
        sigma_loss=1.0,
        fixed_color_sigma=False,
    ):
        """Initialize the moment matching loss.

        Args:
            distribution (str, optional): Type of distribution to use ("gaussian").
                Defaults to "gaussian".
            alpha_f (function, optional): Alpha loss function. Defaults to None.
            alpha_entropy (float, optional): Weight for alpha entropy regularization.
                Defaults to 0.0.
            alpha_smooth (float, optional): Weight for alpha smoothness regularization.
                Defaults to 0.0.
            sigma_loss (float, optional): Weight for sigma loss. Defaults to 1.0.
        """
        super().__init__()
        self.ssim = SSIM()
        self.alpha_f = alpha_f
        self.alpha_entropy = alpha_entropy
        self.alpha_smooth = alpha_smooth
        self.sigma_loss = sigma_loss
        self.fixed_color_sigma = fixed_color_sigma
        if fixed_color_sigma:
            self.fixed_color_sigma = nn.Parameter(torch.tensor(0.0))

    def forward(self, y, mu, sigma, alpha, w_grad, mask):
        """Compute the moment matching loss.

        Combines SSIM and L1 loss with uncertainty weighting and
        additional regularization terms for alpha and sigma.

        Args:
            y (torch.Tensor): Target image, shape [B, C, H, W].
            mu (torch.Tensor): Mean of the mixture, shape [B, D, T, C, H, W].
            sigma (torch.Tensor): Standard deviation of the mixture, shape [B, D, T, C, H, W].
            alpha (torch.Tensor): Mixture weights, shape [B, D, 1, 1, H, W].
            w_grad (torch.Tensor): Gradient weighting, used for edge-aware smoothness.
            mask (torch.Tensor): Validity mask.

        Returns:
            tuple:
                - loss (torch.Tensor): Total loss value.
                - losses (dict): Dictionary of individual loss components.
                - images (dict): Dictionary of images for visualization.
        """
        # fuse dimensions since AvgPool2d inside SSIM can only handle 4D tensors
        B, D, T, C, H, W = mu.shape
        mu = mu.view(B, D * T * C, H, W)
        sigma = sigma.view(B, D * T * C, H, W)
        alpha = alpha.view(B, D, H, W)

        y = y.view(B, 1, 1, C, H, W).repeat(1, D, T, 1, 1, 1).view(B, D * T * C, H, W)

        ssim, ssim_sigma = self.ssim(y, mu, sigma)

        l1 = (y - mu).abs()
        loss = 0.85 * ssim + 0.15 * l1
        sigma = sigma.view(B, D * T * C, H, W)
        if self.fixed_color_sigma:
            loss_sigma = torch.exp(self.fixed_color_sigma) * torch.ones_like(sigma)
        else:
            loss_sigma = 0.85 * ssim_sigma + 0.15 * sigma

        if D > 1:
            lp = best_prob(
                loss.view(B, D, T * C, H, W).detach(), loss_sigma.clip(1e-6).view(B, D, T * C, H, W).detach()
            )

            lpa = alpha.detach().view(B, D, 1, H, W) * lp + 1e-6
            lp_s = lpa.sum(1, True)
            w = lpa / lp_s
            w = w.view(B, D * T * C, H, W)
            alpha_loss = (
                mask.view(B, D, T, 1, H, W).all(1, True)
                * w_grad.view(B, D, 1, 1, H, W)
                * self.alpha_f(alpha.view(B, D, 1, 1, H, W), w.view(B, D, T, C, H, W))
            ).view(B, D, T, C, H, W)
            a = alpha.detach().view(B, D, 1, 1, H, W).round()
            w = w.view(B, D, T, C, H, W)[:, 1]
        else:
            a = 1
            alpha_loss = 0
            w = torch.ones(B, T, C, H, W, device=loss.device)

        mu_loss = loss.view(B, D, T, C, H, W) * a

        if self.sigma_loss > 0:
            sigma_loss = self.sigma_loss * (loss_sigma - loss.detach()).square().view(B, D, T, C, H, W) * a
        else:
            sigma_loss = 0

        if self.alpha_entropy > 0:
            entropy = -alpha * (alpha + 1e-6).log()
            alpha_loss += self.alpha_entropy * entropy.view(B, D, 1, 1, H, W)

        if self.alpha_smooth > 0:
            smooth = spatial_gradient(alpha, "diff").abs().mean(2)
            alpha_loss += self.alpha_smooth * smooth.view(B, D, 1, 1, H, W)

        loss = mu_loss + sigma_loss + alpha_loss * 0.1
        return loss.view(B, D, T, C, H, W), l1.view(B, D, T, C, H, W).detach(), w


def best_prob(mu, sigma):
    """Compute the best probability for each component based on loss and uncertainty.

    Args:
        mu (torch.Tensor): Mean values (or loss values), shape [B, D, C, H, W].
        sigma (torch.Tensor): Standard deviation values, shape [B, D, C, H, W].

    Returns:
        torch.Tensor: Probability values for each component, shape [B, D, C, H, W].
    """
    diff_mu = mu[:, 1] - mu[:, 0]
    diff_sigma_sq = sigma[:, 1] ** 2 + sigma[:, 0] ** 2

    # Compute the probability that the second depth is better than the first
    # using the error function (erf) for the Gaussian distribution
    prob = 0.5 * (1 - torch.special.erf(diff_mu / torch.sqrt(2 * diff_sigma_sq)))
    return torch.stack([1 - prob, prob], dim=1)
