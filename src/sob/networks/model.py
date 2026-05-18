import os
import timm
import torch
import torch.nn as nn
from .depth_decoder import DepthDecoder
from .hr_decoder import HRDepthDecoder
from .pose_decoder import PoseDecoder
from .resnet_encoder import ResnetEncoder


class DepthModel(nn.Module):
    """Depth estimation model with encoder-decoder architecture.

    This model uses either a ResNet or a model from the timm library as the encoder
    and a custom depth decoder that produces depth distribution parameters.

    Attributes:
        encoder: Encoder network that extracts features from an input image.
        decoder: Decoder network that predicts depth from features.
    """

    def __init__(
        self, encoder, num_layers=None, components=2, decoder="depth", sigma_type="learned_dual", pretrain_path=None
    ):
        """Initialize the depth model.

        Args:
            encoder (str): Encoder architecture name when using timm models.
            num_layers (int, optional): Number of layers for ResNet encoder.
                If None, uses timm model. Defaults to None.
            components (int, optional): Number of mixture components for depth
                distribution. Defaults to 2.
            decoder (str, optional): Type of decoder to use. One of "depth", "hr".
                Defaults to "depth".
        """
        super().__init__()

        self.standardize = Standardize()
        if num_layers is None:
            self.encoder = timm.create_model(encoder, features_only=True, pretrained=False)
            if pretrain_path is not None:
                load_convnext_weights(self.encoder, pretrain_path)
        else:
            self.encoder = ResnetEncoder(num_layers, pretrained=pretrain_path is not None)
            self.encoder.feature_info = [
                {"num_chs": 64, "reduction": 2},
                {"num_chs": 64, "reduction": 4},
                {"num_chs": 128, "reduction": 8},
                {"num_chs": 256, "reduction": 16},
                {"num_chs": 512, "reduction": 32},
            ]

        if decoder == "hr":
            self.decoder = HRDepthDecoder(self.encoder.feature_info, components=components, sigma_type=sigma_type)
        else:
            self.decoder = DepthDecoder(self.encoder.feature_info, components=components, sigma_type=sigma_type)

    def forward(self, x):
        """Forward pass through the depth model.

        Args:
            x (torch.Tensor): Input image tensor, shape [B, 3, H, W].

        Returns:
            dict: Dictionary containing depth distribution parameters at different scales.
        """
        x = self.standardize(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class PoseModel(nn.Module):
    """Pose estimation model with encoder-decoder architecture.

    This model takes a pair of images and estimates the relative pose (rotation and translation)
    between them using a shared encoder and pose decoder.

    Attributes:
        encoder: Encoder network that extracts features from a pair of images.
        decoder (PoseDecoder): Decoder network that predicts pose from features.
    """

    def __init__(self, num_layers=None, pretrain_path=None):
        """Initialize the pose model.

        Args:
            num_layers (int, optional): Number of layers for ResNet encoder.
                If None, uses timm model. Defaults to None.
            pretrain_path (str, optional): Path to pretrained weights. Defaults to None.
        """
        super().__init__()
        self.standardize = Standardize(n_images=2)
        if num_layers is None:
            self.encoder = timm.create_model(
                "resnet18", pretrained=False, features_only=True, out_indices=(0, 1, 2, 3, 4), in_chans=6
            )
            if pretrain_path is not None:
                load_resnet_weights(self.encoder, pretrain_path)
            num_chs = self.encoder.feature_info[-1]["num_chs"]
        else:
            self.encoder = ResnetEncoder(num_layers, pretrained=pretrain_path is not None, num_input_images=2)
            num_chs = 512

        self.decoder = PoseDecoder(num_chs)

    def forward(self, x):
        """Forward pass through the pose model.

        Args:
            x (torch.Tensor): Input tensor of concatenated source and target images,
                shape [B, 6, H, W].

        Returns:
            tuple: Tuple of (axisangle, translation) tensors.
        """
        x = self.standardize(x)
        features = self.encoder(x)
        pose = self.decoder(features[-1])
        return pose


class Standardize(nn.Module):
    """Standardize the input tensor to have zero mean and unit variance.

    This module computes the mean and standard deviation of the input tensor
    and applies a transformation to standardize it.

    Attributes:
        mean (torch.Tensor): Mean value for each channel.
        std (torch.Tensor): Standard deviation for each channel.
    """

    def __init__(self, n_images=1):
        super().__init__()
        self.mean = nn.Parameter(
            torch.tensor((0.485, 0.456, 0.406)).repeat(n_images).view(1, -1, 1, 1), requires_grad=False
        )
        self.std = nn.Parameter(
            torch.tensor((0.229, 0.224, 0.225)).repeat(n_images).view(1, -1, 1, 1), requires_grad=False
        )

    def forward(self, x):
        """Forward pass through the standardization module.

        Args:
            x (torch.Tensor): Input tensor to be standardized.

        Returns:
            torch.Tensor: Standardized tensor.
        """
        return (x - self.mean) / self.std


def load_resnet_weights(encoder, pretrain_path):
    """Load pretrained weights into the model.

    Args:
        model (nn.Module): The model to load weights into.
        pretrained_path (str): Path to the pretrained weights file.
    """
    tensors = torch.load(os.path.join(pretrain_path, "resnet18-5c106cde.pth"), weights_only=True)

    def expand_conv_weight(state_dict, target_in_channels=6):
        """
        Expand the first convolution weight from state_dict to have target_in_channels.
        Assumes original weight shape is (out_channels, in_channels, kH, kW)
        and that original in_channels is half of target_in_channels (e.g. 3 -> 6).
        """
        key = "conv1.weight"
        if key in state_dict:
            weight = state_dict[key]
            current_in_channels = weight.size(1)
            if current_in_channels == target_in_channels:
                return state_dict
            elif current_in_channels * 2 == target_in_channels:
                # Repeat the weights along the channel dimension and average.
                # This operation doubles the channels from 3 to 6.
                state_dict[key] = weight.repeat(1, 2, 1, 1) / 2.0
        return state_dict

    # remove fc layers
    tensors = {k: v for k, v in tensors.items() if "fc." not in k}
    # Expand first conv weight from 3 to 6 channels if needed.
    renamed = expand_conv_weight(tensors, target_in_channels=6)
    encoder.load_state_dict(renamed, strict=True)


def load_convnext_weights(encoder, pretrain_path):
    """Load pretrained weights into the model.

    Args:
        model (nn.Module): The model to load weights into.
        pretrained_path (str): Path to the pretrained weights file.
    """
    tensors = torch.load(os.path.join(pretrain_path, "convnext_base_1k_224_ema.pth"), weights_only=True)

    def rename_keys(ckpt):
        result = {}
        model_dict = ckpt["model"]

        for key, value in model_dict.items():
            # Skip head weights which we don't need
            if "head" in key or "norm" in key and len(key.split(".")) == 2:
                continue

            # Handle stem (first downsample layers)
            if key.startswith("downsample_layers.0"):
                parts = key.split(".")
                new_key = f"stem_{parts[2]}.{parts[3]}" if len(parts) > 3 else f"stem_{parts[2]}"

            # Handle other downsample layers
            elif key.startswith("downsample_layers"):
                parts = key.split(".")
                stage_num = parts[1]
                new_key = (
                    f"stages_{stage_num}.downsample.{parts[2]}.{parts[3]}"
                    if len(parts) > 3
                    else f"stages_{stage_num}.downsample.{parts[2]}"
                )

            # Handle stages
            elif key.startswith("stages"):
                parts = key.split(".")
                if len(parts) >= 3:
                    stage_num = parts[1]
                    block_num = parts[2]

                    # Convert component names
                    if len(parts) > 3:
                        component = parts[3]
                        if component == "dwconv":
                            component = "conv_dw"
                        elif component == "pwconv1":
                            component = "mlp.fc1"
                        elif component == "pwconv2":
                            component = "mlp.fc2"

                        new_key = f"stages_{stage_num}.blocks.{block_num}.{component}"
                        if len(parts) > 4:
                            new_key += f".{parts[4]}"
                    else:
                        new_key = f"stages_{stage_num}.blocks.{block_num}"
            else:
                new_key = key

            result[new_key] = value

        return result

    renamed_state_dict = rename_keys(tensors)

    encoder.load_state_dict(renamed_state_dict, strict=True)
