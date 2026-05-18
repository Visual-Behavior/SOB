import torch
from torch import nn


class AutoMask(nn.Module):
    """Auto-masking filter to handle static pixels in self-supervised depth estimation.

    This module implements the auto-masking technique from Monodepth2, which filters out
    pixels that have better photometric error when using the identity transformation
    compared to using the estimated depth and pose. This helps handle static objects.

    Attributes:
        min_filter (str): Strategy for computing minimum reprojection error
            ("default", "global", or "alpha").
        sources (list): Source frame types for self-supervision.
    """

    def __init__(self, min_filter):
        """Initialize the auto-masking filter.

        Args:
            min_filter (str): Strategy for computing minimum reprojection error
                ("default", "global", or "alpha").
        """
        super().__init__()
        self.min_filter = min_filter

    def forward(self, error, l1, inputs: dict, alpha=None, mask=None):
        """Automasking with mixtures.

        Args:
            error: Tensor of shape [B, D, T, C, H, W] representing the error.
            l1: Tensor of shape [B, D, T, H, W] representing the L1 loss.
            inputs: Dictionary containing input images.
            alpha: Optional tensor of shape [B, D, 1, H, W].
            mask: Optional tensor of shape [B, D, T, H, W].

        Returns:
            Filtered error tensor of shape [B, 1, H, W].
        """

        sources = torch.stack([inputs[f"source_{source}"] for source in self.sources], 1)  # B T C H W
        identity = (inputs[f"target"][:, None] - sources).abs().mean(2).amin(1, True)  # B, 1, H, W
        l1 = l1.mean(3)  # B, D, T, H, W

        if self.min_filter == "global":
            B, D, T, H, W = l1.shape
            l1_min, l1_amin = l1.view(B, D * T, H, W).min(dim=1, keepdim=True)  # (B, 1, H, W)
            l1_amin = (l1_amin % T).unsqueeze(1).expand(B, D, 1, H, W)  # same T for each D
            l1_min = l1_min.unsqueeze(1).expand(B, D, 1, H, W)

        elif self.min_filter == "alpha":
            B, D, T, H, W = l1.shape
            l1_alpha = (l1 * alpha.detach().round().unsqueeze(2)).sum(2, True)  # B D 1 H W
            l1_min, l1_amin = l1_alpha.min(2, keepdim=True)  # B 1 1 H W

        else:
            l1_min, l1_amin = l1.min(2, keepdim=True)  # l1_m and l1_am: (B, D, 1, H, W)

        id_mask = l1_min < identity.unsqueeze(1)  # (B, D, 1, H, W)

        error = error.mean(3)  # B, D, T, H, W

        if mask is not None:
            error = error * mask.squeeze(3)  # B, D, T, H, W

        return (id_mask * error.take_along_dim(l1_amin, 2)).sum(1)  # B 1 H W
