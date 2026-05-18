import torch
import torch.nn as nn
from collections import OrderedDict


class ConvBlock(nn.Module):
    """Convolution block with ELU activation.

    This module performs a 3x3 convolution followed by an ELU activation function.
    It's a basic building block used throughout the network architecture.

    Attributes:
        conv (Conv3x3): 3x3 convolution layer with padding.
        nonlin (nn.ELU): ELU activation function.
    """

    def __init__(self, in_channels, out_channels):
        """Initialize the ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        """Perform forward pass through the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after convolution and activation.
        """
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """3x3 convolution with padding.

    This module performs a 3x3 convolution with reflection or zero padding.
    It's used as a basic building block in the network architecture.

    Attributes:
        pad (nn.Module): Reflection or zero padding layer.
        conv (nn.Conv2d): 3x3 convolution layer.
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        """Initialize the Conv3x3 layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_refl (bool, optional): If True, uses reflection padding.
                Otherwise uses zero padding. Defaults to True.
        """
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        """Perform forward pass through the Conv3x3 layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after padding and convolution.
        """
        out = self.pad(x)
        out = self.conv(out)
        return out


def conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    """Layer to convolve input."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=bias)


def conv3x3(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    """Layer to pad and convolve input."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1, padding_mode="reflect", bias=bias)


def conv_block(in_ch: int, out_ch: int) -> nn.Module:
    """Layer to perform a convolution followed by ELU."""
    return nn.Sequential(
        OrderedDict(
            {
                "conv": conv3x3(in_ch, out_ch),
                "act": nn.ELU(inplace=True),
            }
        )
    )


def clip_exp(x, min=-10, max=10):
    """Clip the exponential of x such that exp(x) is not nan or inf.

    Args:
        x (torch.Tensor): Input tensor.
        min_val (float, optional): Minimum value for clipping. Defaults to 1e-6.
        max_val (float, optional): Maximum value for clipping. Defaults to 1e6.

    Returns:
        torch.Tensor: Clipped tensor.
    """
    return torch.exp(torch.clamp(x, min=min, max=max))
