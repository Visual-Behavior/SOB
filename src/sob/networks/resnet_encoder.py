import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet18_Weights, ResNet50_Weights

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class ResNetMultiImageInput(models.ResNet):
    """ResNet model modified to accept multiple stacked images as input.

    This class extends the standard ResNet to support multiple concatenated images
    as input, which is useful for tasks that require temporal information like
    pose estimation from consecutive frames.

    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    Attributes:
        All attributes from ResNet plus:
        conv1 (nn.Conv2d): Modified first convolution to accept multiple stacked images.
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        """Initialize ResNetMultiImageInput.

        Args:
            block (nn.Module): ResNet block type (BasicBlock or Bottleneck).
            layers (list): Number of blocks in each layer.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            num_input_images (int, optional): Number of images stacked as input. Defaults to 1.
        """
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Construct a ResNet model with multi-image input support.

    Args:
        num_layers (int): Number of ResNet layers. Must be 18 or 50.
        pretrained (bool, optional): If True, initializes with ImageNet pretrained weights.
            Defaults to False.
        num_input_images (int, optional): Number of frames stacked as input.
            Defaults to 1.

    Returns:
        ResNetMultiImageInput: ResNet model configured for multiple input images.

    Raises:
        AssertionError: If num_layers is not 18 or 50.
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = load_state_dict_from_url(model_urls["resnet{}".format(num_layers)])
        loaded["conv1.weight"] = torch.cat([loaded["conv1.weight"]] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """ResNet-based encoder for feature extraction.

    This class wraps a ResNet model for use as a feature encoder in depth and pose
    estimation networks. It provides hierarchical features from different ResNet layers.

    Attributes:
        num_ch_enc (numpy.ndarray): Number of channels in each encoder layer.
        encoder (nn.Module): ResNet model used for encoding.
        features (list): Feature maps from each layer during forward pass.
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        """Initialize the ResNet encoder.

        Args:
            num_layers (int): Number of ResNet layers (18, 34, 50, 101, or 152).
            pretrained (bool): Whether to use pretrained ImageNet weights.
            num_input_images (int, optional): Number of input images concatenated
                channel-wise. Defaults to 1.

        Raises:
            ValueError: If num_layers is not a valid ResNet size.
        """
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        weights = {18: ResNet18_Weights, 50: ResNet50_Weights}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](weights=None if not pretrained else weights[num_layers].DEFAULT)

        # Remove the final classification layer
        del self.encoder.fc

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        """Extract hierarchical features from input image.

        Args:
            input_image (torch.Tensor): Input image tensor, shape [B, 3*num_input_images, H, W].

        Returns:
            list: List of feature tensors from each layer of the encoder,
                ordered from low to high level features.
        """
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
