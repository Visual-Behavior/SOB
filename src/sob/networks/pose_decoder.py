import torch.nn as nn


class PoseDecoder(nn.Module):
    """Decoder network for pose estimation.

    This network processes encoder features to estimate the relative camera pose
    (rotation and translation) between two frames.

    Attributes:
        pose_net (nn.Sequential): Sequence of convolutional layers for pose prediction.
    """

    def __init__(self, num_ch_features):
        """Initialize the pose decoder.

        Args:
            num_ch_features (int): Number of input channels from the encoder.
        """
        super(PoseDecoder, self).__init__()

        self.pose_net = nn.Sequential(
            self.block(num_ch_features, 256, kernel_size=1),
            self.block(256, 256, kernel_size=3, padding=1),
            self.block(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 6, kernel_size=1),
        )

    @staticmethod
    def block(in_ch: int, out_ch: int, kernel_size: int, padding: int = 0) -> nn.Module:
        """Conv + ReLU."""
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding), nn.ReLU(inplace=True))

    def forward(self, features):
        """Process encoder features to predict relative camera pose.

        Args:
            features (torch.Tensor): Encoder features, shape [B, C, H, W].

        Returns:
            tuple:
                - axisangle (torch.Tensor): Rotation in axis-angle representation, shape [B, 3].
                - translation (torch.Tensor): Translation vector, shape [B, 3].
        """
        out = 0.01 * self.pose_net(features).mean(dim=(2, 3))
        return out[:, :3], out[:, 3:]  # R, t
