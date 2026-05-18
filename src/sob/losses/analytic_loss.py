import torch
from torch import nn
from torch.nn import functional as F

from ..config import TrainingConfig
from ..sampling import bilinear_sample
from ..projection import static_flow, static_flow_inv
from ..utils import get_transform_matrix
from .component_loss import MomentMatchingLoss
from .utils import cross_entropy, mse, mae, attn
from .mask import AutoMask


class AnalyticLoss(nn.Module):
    """Analytical reprojection loss for self-supervised depth estimation.

    This loss implements an analytical version of the reprojection loss that
    can propagate uncertainty from depth to reprojection. It can use different
    component losses and alpha loss functions.

    Attributes:
        component_loss: Loss function for depth distribution components.
        sources (tuple): Source frame types for self-supervision.
        flow (function): Function for computing optical flow.
        HW (tuple): (height, width) dimensions.
        auto_mask (AutoMask): Auto-masking module.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the analytic loss.

        Args:
            config (TrainingConfig): Configuration object with loss settings.
        """
        super().__init__()

        if config.alpha_loss == "ce":
            alpha_f = cross_entropy
        elif config.alpha_loss == "mse":
            alpha_f = mse
        elif config.alpha_loss == "mae":
            alpha_f = mae
        elif config.alpha_loss == "attn":
            alpha_f = attn

        self.component_loss = MomentMatchingLoss(
            config.distribution,
            alpha_f,
            config.alpha_entropy,
            config.alpha_smooth,
            config.sigma_loss * (config.sigma_type != "none"),
            config.sigma_type == "fixed_color",
        )

        self.sources = config.sources
        self.flow = static_flow if config.no_inv else static_flow_inv
        self.HW = config.HW
        self.auto_mask = AutoMask(config.filter_min)
        self.auto_mask.sources = self.sources
        self.fixed_sigma_color = config.sigma_type == "fixed_color"
        self.inv = not config.no_inv
        self.sampler = bilinear_sample

    def reproject(self, inputs, mu, sigma, pose_output):
        """Reproject target pixels to source views using depth and pose.

        For each pixel in the target image, this method:
        1. Computes the pixel flow using depth and pose
        2. Samples colors from the source images at the warped locations
        3. Returns the means, standard deviations, and validity masks

        Args:
            inputs (dict): Dictionary containing input data including source images.
            mu (torch.Tensor): Mean depth values, shape [B, 1, H, W].
            sigma (torch.Tensor): Depth standard deviation values, shape [B, 1, H, W].
            pose_output (dict): Dictionary of pose outputs for each source.

        Returns:
            tuple:
                - mu_c (torch.Tensor): Mean of the sampled colors, shape [B, T, C, H, W].
                - sigma_c (torch.Tensor): Standard deviation of the sampled colors, shape [B, T, C, H, W].
                - mask (torch.Tensor): Mask of valid pixels, shape [B, T, 1, H, W].
        """
        outputs = []
        source_flows = {}
        source_transforms = {}

        # First pass: calculate flows and transforms for all sources
        for source in self.sources:
            T = get_transform_matrix(*pose_output[source], invert=(source < 0))
            source_transforms[source] = T
            K_target = inputs["K"]
            mu_pos, sigma_pos = self.flow(mu, sigma, T, K_target, inputs["K"])
            source_flows[source] = (mu_pos, sigma_pos)

        # Second pass: reproject the source images to the target reference frame
        for source in self.sources:
            mu_pos, sigma_pos = source_flows[source]
            mu_i, sigma_i, mask_i = self.sampler(mu_pos, sigma_pos, inputs[f"source_{source}"])

            outputs.append((mu_i, sigma_i, mask_i))

        mu_c, sigma_c, mask = (torch.stack(x, 1) for x in zip(*outputs))  # B T C H W
        return mu_c, sigma_c, mask

    def forward(self, inputs, depth_output, pose_output):
        """Compute the analytical reprojection loss.

        This method:
        1. Processes each depth scale
        2. Reprojects depth to source views
        3. Computes auto-masking
        4. Computes the component loss

        Args:
            inputs (dict): Dictionary containing input data.
            depth_output (list): List of depth outputs at different scales.
                Each element is a dictionary with keys 'mu', 'sigma', 'alpha'.
            pose_output (dict): Dictionary of pose outputs for each source.

        Returns:
            tuple:
                - loss (torch.Tensor): Total loss value.
                - losses (dict): Dictionary of individual loss components.
                - images (dict): Dictionary of images for visualization.
        """
        loss = 0
        images = {}
        # iterate over the scales
        for i, mixture in enumerate(depth_output):
            mu, sigma, alpha = mixture["mu"], mixture["sigma"], mixture["alpha"]

            mu = F.interpolate(mu, self.HW, mode="bilinear", align_corners=False)
            sigma = F.interpolate(sigma, self.HW, mode="bilinear", align_corners=False)

            outputs = []
            for k in range(mu.shape[1]):  # Iterate over all components
                mu_k, sigma_k = mu[:, k : k + 1], sigma[:, k : k + 1]
                outputs.append(self.reproject(inputs, mu_k, sigma_k, pose_output))
            mu_c, sigma_c, mask = (torch.stack(x, 1) for x in zip(*outputs))  # B D T C H W

            alpha = F.interpolate(alpha, self.HW, mode="bilinear", align_corners=False)  # B D 1 H W
            target = inputs["target"]

            w = torch.ones_like(mu)

            error, l1, w = self.component_loss(target, mu_c, sigma_c, alpha, w, mask)
            error = self.auto_mask(error, l1, inputs, alpha, mask)
            loss += error.mean()

            if i == 0:
                images["mu_c"] = (mu_c * mask).mean([1, 2])
                images["error"] = error.detach()
                images["debug"] = (mu_c, sigma_c, alpha)
                images["w"] = w.detach().mean([1, 2]).unsqueeze(1)
                images["mask"] = mask.detach().mean([1, 2])

        loss = loss / len(depth_output)
        return loss, {"reproj_loss": loss.detach()}, images
