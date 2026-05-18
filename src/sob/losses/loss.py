import torch
from torch import nn
from ..config import TrainingConfig
from .smoothness import SmoothnessLoss
from .analytic_loss import AnalyticLoss


class GeometricLoss(nn.Module):
    """Main loss function for self-supervised depth and pose estimation.

    This class orchestrates multiple loss components for training:
    1. Reprojection loss
    2. Smoothness loss (optional)

    Attributes:
        config (TrainingConfig): Configuration object with loss settings.
        reproj_loss: Reprojection loss module chosen based on config.
        use_smoothness_loss (bool): Whether to use depth smoothness loss.
        smoothness_loss (SmoothnessLoss, optional): Loss for depth smoothness.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the geometric loss.

        Args:
            config (TrainingConfig): Configuration object with loss settings.
        """
        super().__init__()
        self.config = config
        self.reproj_loss = AnalyticLoss(config)
        self.smoothness_loss = SmoothnessLoss(config.smoothness, config.smoothness_sigma)

    def forward(self, inputs, depth_output, pose_output):
        """Compute the total loss for training.

        Args:
            inputs (dict): Dictionary of input tensors including target and source images.
            depth_output (dict): Dictionary of depth outputs from the depth model.
            pose_output (dict): Dictionary of pose outputs from the pose model.

        Returns:
            tuple:
                - loss (torch.Tensor): Total loss value.
                - reproj_logs (dict): Dictionary of loss components for logging.
                - reproj_images (dict): Dictionary of images for visualization.
        """

        loss, scalar_logs, image_logs = self.reproj_loss(inputs, depth_output, pose_output)

        sigma_smoothness_loss = self.smoothness_loss(depth_output, inputs["target"])
        scalar_logs["smoothness"] = sigma_smoothness_loss.detach()
        loss += sigma_smoothness_loss

        return loss, scalar_logs, image_logs
