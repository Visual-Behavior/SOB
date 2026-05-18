import torch
from kornia.morphology import erosion


def bilinear_sample(mu, sigma, image):
    """Bilinear interpolation of image with uncertainty propagation.

    Performs bilinear interpolation of the sampled pixels with standard
    deviation estimate with align_corners=True.

    Args:
        mu (torch.Tensor): Mean positions, shape [B, 2, H, W].
        sigma (torch.Tensor, optional): Standard deviation of positions, shape [B, 2, H, W].
            If None, only the means are computed.
        image (torch.Tensor): Source RGB image, shape [B, 3, H, W].

    Returns:
        tuple: If sigma is not None:
            - mu_c (torch.Tensor): Mean of sampled pixels, shape [B, 3, H, W].
            - sigma_c (torch.Tensor): Standard deviation of sampled pixels, shape [B, 3, H, W].
            - mask (torch.Tensor): Valid sampling mask, shape [B, 1, H, W].
          If sigma is None:
            - mu_c (torch.Tensor): Mean of sampled pixels, shape [B, 3, H, W].
            - mask (torch.Tensor): Valid sampling mask, shape [B, 1, H, W].
    """
    B, _, H, W = mu.shape
    # remove pixels that are taken out of bounds
    mask = (mu[:, :1] > 0) & (mu[:, 1:] > 0) & (mu[:, :1] < (W - 1)) & (mu[:, 1:] < (H - 1))
    mask_e = erosion(mask.float(), torch.ones(3, 3, device=mu.device))
    # use erosion since SSIM computes loss with neighboring pixels
    x0 = torch.floor(mu.detach())
    x1 = x0 + 1
    x0 = (x0 * mask).long()
    x1 = (x1 * mask).long()
    ind = torch.arange(B, device=image.device)
    ind = ind.view(B, 1, 1).expand(-1, H, W).long()
    # x(u,v)
    # x00 ---- x10
    #  |        |
    #  |        |
    # x01 ---- x11
    # - B 3 H W -
    c00 = image[ind, :, x0[:, 1], x0[:, 0]].permute(0, 3, 1, 2)
    c01 = image[ind, :, x1[:, 1], x0[:, 0]].permute(0, 3, 1, 2)
    c10 = image[ind, :, x0[:, 1], x1[:, 0]].permute(0, 3, 1, 2)
    c11 = image[ind, :, x1[:, 1], x1[:, 0]].permute(0, 3, 1, 2)

    k00 = c00
    k01 = c01 - c00
    k10 = c10 - c00
    k11 = c11 + c00 - c01 - c10

    mu = mu - x0
    mu_x, mu_y = mu[:, :1], mu[:, 1:]
    mu_c = mu_x * mu_y * k11 + mu_x * k10 + mu_y * k01 + k00

    if sigma is not None:
        sigma_x, sigma_y = sigma[:, :1], sigma[:, 1:]
        sigma_c = (sigma_x * mu_y.detach() + sigma_y * mu_x.detach()) * k11 + sigma_x * k10 + sigma_y * k01
        return mu_c, sigma_c.clip(1e-6), mask_e

    return mu_c, mask_e
