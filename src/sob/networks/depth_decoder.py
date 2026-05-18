import torch
import torch.nn as nn
from .layers import ConvBlock, Conv3x3, clip_exp
import torch.nn.functional as F
from functools import partial


class Decoder(nn.Module):
    """Decoder network for depth estimation.

    This decoder takes features from an encoder network and produces depth
    distribution parameters (mu, sigma, alpha) for each scale. It uses
    skip connections from the encoder and upsampling to recover spatial details.

    Attributes:
        components (int): Number of mixture components for depth distribution.
        output_channels (int): Number of output channels (components*3-1).
        scales (range): Range of scales to output.
        num_ch_enc (list): Number of channels in encoder features.
        enc_scales (list): Reduction factors for encoder features.
        num_ch_dec (list): Number of channels in decoder features.
        dec_scales (list): Scale factors for decoder features.
        first_feature_idx (int): Index of first encoder feature to use.
        upsample (function): Upsampling function.
        conv0 (nn.ModuleList): First convolution block at each decoder level.
        conv1 (nn.ModuleList): Second convolution block at each decoder level.
        out_conv (nn.ModuleList): Output convolution for each scale.
    """

    def __init__(
        self,
        features_info,
        scales=range(4),
        out_channels=1,
    ):
        """Initialize the depth decoder.

        Args:
            features_info (list): List of dictionaries containing encoder features info
                with keys 'num_chs' (number of channels) and 'reduction' (spatial reduction factor).
            scales (range, optional): Scales to output depth at. Defaults to range(4).
            components (int, optional): Number of mixture components. Defaults to 2.
        """
        super().__init__()
        self.output_channels = out_channels
        self.scales = scales

        self.num_ch_enc = [fi["num_chs"] for fi in features_info]
        self.enc_scales = [fi["reduction"] for fi in features_info]
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.dec_scales = [2**i for i in range(len(self.num_ch_dec))]
        self.first_feature_idx = self.dec_scales.index(self.enc_scales[0])

        self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest-exact")

        conv0_in_ch = [self.num_ch_dec[i] for i in range(1, len(self.num_ch_dec))] + [self.num_ch_enc[-1]]
        self.conv0 = nn.ModuleList([ConvBlock(in_ch, self.num_ch_dec[i]) for i, in_ch in enumerate(conv0_in_ch)])
        conv1_in_ch = self.num_ch_dec.copy()  # take previous layer as input
        for i in range(len(self.num_ch_dec)):
            if 2**i in self.enc_scales:  # skip connection if there is a corresponding features
                conv1_in_ch[i] += self.num_ch_enc[self.enc_scales.index(2**i)]

        self.conv1 = nn.ModuleList([ConvBlock(in_ch, self.num_ch_dec[i]) for i, in_ch in enumerate(conv1_in_ch)])
        self.out_conv = nn.ModuleList([Conv3x3(self.num_ch_dec[s], self.output_channels) for s in self.scales])

    def _decode_layer(self, i, x, skip_features):
        """Process a single decoder layer.

        Args:
            i (int): Decoder layer index.
            x (torch.Tensor): Input features from previous layer.
            skip_features (torch.Tensor): Skip connection features from encoder.

        Returns:
            torch.Tensor: Processed features.
        """
        x = self.upsample(self.conv0[i](x))

        if i >= self.first_feature_idx:
            x = torch.cat([x, skip_features], 1)

        return self.conv1[i](x)

    def _output_layer(self, scale, x):
        """Output raw logits. This function should likely be overridden by the subclass.

        Args:
            scale (int): Scale index.
            x (torch.Tensor): Decoder features.

        Returns:
            torch.Tensor: Raw logits.
        """
        return self.out_conv[scale](x)

    def forward(self, input_features):
        """Forward pass for the decoder.

        This method processes encoder features through the decoder network
        to produce outputs at multiple scales.

        Args:
            input_features (list): List of feature tensors from the encoder,
                ordered from highest resolution to lowest resolution.

        Returns:
            list: List of outputs, ordered from lowest resolution to highest resolution.
        """
        outputs = []
        x = self._decode_layer(4, input_features[-1], input_features[-2])

        # go from low resolution to high resolution
        for i in range(3, -1, -1):
            x = self._decode_layer(i, x, input_features[i - self.first_feature_idx])
            outputs.append(self._output_layer(i, x))

        return outputs[::-1]  # High resolution to low resolution

# Example with Resnet18 encoder

# High Res
# Scale 1   Features
#                          Conv1[0] --> OutConv[0] -->
#                             ^
#                             |
#                     Conv0[0]+Upsample
#                             ^
#                             |
# Scale 2   [F0] --------> Conv1[1] --> OutConv[1] -->
#                             ^
#                             |
#                     Conv0[1]+Upsample
#                             ^
#                             |
# Scale 4   [F1] --------> Conv1[2] --> OutConv[2] -->
#                             ^
#                             |
#                     Conv0[2]+Upsample
#                             ^
#                             |
# Scale 8   [F2] --------> Conv1[3] --> OutConv[3] -->
#                             ^
#                             |
#                     Conv0[3]+Upsample
#                             ^
#                             |
# Scale 16  [F3] --------> Conv1[4]
#                             ^
#                             |
#                     Conv0[4]+Upsample
#                             ^
#                             |
# Scale 32  [F4] ------------ +
# Low Res


class DepthDecoder(Decoder):
    """Decoder network for depth estimation.

    This decoder takes features from an encoder network and produces depth
    distribution parameters (mu, sigma, alpha) for each scale. It uses
    skip connections from the encoder and upsampling to recover spatial details.

    Attributes:
        components (int): Number of mixture components for depth distribution.
        output_channels (int): Number of output channels (components*3-1).
        scales (range): Range of scales to output.
        num_ch_enc (list): Number of channels in encoder features.
        enc_scales (list): Reduction factors for encoder features.
        num_ch_dec (list): Number of channels in decoder features.
        dec_scales (list): Scale factors for decoder features.
        first_feature_idx (int): Index of first encoder feature to use.
        upsample (function): Upsampling function.
        conv0 (nn.ModuleList): First convolution block at each decoder level.
        conv1 (nn.ModuleList): Second convolution block at each decoder level.
        out_conv (nn.ModuleList): Output convolution for each scale.
    """

    def __init__(
        self,
        features_info,
        scales=range(4),
        components=2,
        sigma_type="learned_dual",
    ):
        """Initialize the depth decoder.

        Args:
            features_info (list): List of dictionaries containing encoder features info
                with keys 'num_chs' (number of channels) and 'reduction' (spatial reduction factor).
            scales (range, optional): Scales to output depth at. Defaults to range(4).
            components (int, optional): Number of mixture components. Defaults to 2.
        """

        self.components = components
        self.sigma_type = sigma_type
        if sigma_type == "learned_dual":
            assert components == 2, "learned_dual only supports 2 components"
            out_channels = components * 3 - 1
        elif sigma_type == "learned_single":
            out_channels = components * 2
        else:
            out_channels = components * 2 - 1

        super().__init__(features_info, scales, out_channels)

        if sigma_type == "fixed_depth_single":
            self.fixed_depth_sigma = nn.Parameter(torch.tensor(0.0))
        elif sigma_type == "fixed_depth_dual":
            self.fixed_depth_sigma = nn.Parameter(torch.zeros(1, components, 1, 1))

    def _output_layer(self, scale, x):
        """Convert decoder features to depth distribution parameters.

        Args:
            scale (int): Scale index.
            x (torch.Tensor): Decoder features.

        Returns:
            dict: Dictionary with keys 'mu', 'sigma', and 'alpha' containing
                the distribution parameters.
        """
        x = self.out_conv[scale](x)

        output = {
            "mu": F.sigmoid(x[:, : self.components]) * 9.99 + 0.01,
        }
        if self.sigma_type == "learned_dual":
            output["sigma"] = clip_exp(x[:, self.components : self.components * 2])
        elif self.sigma_type == "learned_single":
            output["sigma"] = clip_exp(x[:, self.components : self.components + 1]).repeat(1, self.components, 1, 1)
        elif self.sigma_type == "fixed_depth_single":
            output["sigma"] = clip_exp(self.fixed_depth_sigma) * torch.ones_like(output["mu"])
        elif self.sigma_type == "fixed_depth_dual":
            output["sigma"] = clip_exp(self.fixed_depth_sigma) * torch.ones_like(output["mu"])

        if self.components == 2:
            alpha = x[:, -1:].sigmoid()
            output["alpha"] = torch.cat([1 - alpha, alpha], dim=1)

        return output
