from torchvision.transforms.v2 import ColorJitter as TorchColorJitter
from torchvision.transforms.v2.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    crop,
    resize,
    center_crop,
)
from random import randint, uniform
from torchvision.transforms.v2 import Transform


class ColorJitter(TorchColorJitter):
    """Color jitter augmentation with parameter sampling.

    Extends TorchColorJitter to support sampling and applying the same
    color transformations across multiple images, which is useful for
    consistent augmentation across stereo pairs or video frames.

    Inherits all attributes from TorchColorJitter.
    """

    def sample_params(self):
        """Sample color jitter parameters.

        Returns:
            tuple: Parameters for brightness, contrast, saturation, and hue adjustments
                that can be passed to forward().
        """
        return self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

    def forward(self, img, params):
        """Apply color transformation to an image using pre-sampled parameters.

        This allows applying the same transformation to multiple images.

        Args:
            img (torch.Tensor): Image tensor with shape [..., C, H, W].
            params (tuple): Tuple containing
                (fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor).

        Returns:
            torch.Tensor: Transformed image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = adjust_hue(img, hue_factor)
        return img


class RandomCrop(Transform):
    """Random crop augmentation with camera intrinsics adjustment.

    This transform randomly crops input images and adjusts camera intrinsics
    accordingly, which is important for geometric consistency in depth estimation.

    Attributes:
        HW (tuple): Target crop dimensions (height, width).
        h_margin (float): Vertical margin as a fraction of image height.
        w_margin (float): Horizontal margin as a fraction of image width.
    """

    def __init__(self, HW):
        """Initialize the random crop transform.

        Args:
            HW (tuple): Target crop dimensions (height, width).
        """
        super().__init__()
        self.HW = HW
        self.h_margin = 0.2
        self.w_margin = 0.1

    def sample_params(self, shape, K):
        """Sample crop parameters and adjust camera intrinsics.

        Args:
            shape (tuple): Input image shape.
            K (torch.Tensor): Camera intrinsics matrix.

        Returns:
            tuple:
                - params (tuple): Parameters for cropping (top, left, height, width).
                - K_new (torch.Tensor): Adjusted camera intrinsics for the cropped image.
        """
        *_, H, W = shape

        # Calculate valid regions considering margins
        h_start = int(H * self.h_margin)
        h_end = int(H * (1 - self.h_margin))
        w_start = int(W * self.w_margin)
        w_end = int(W * (1 - self.w_margin))

        # Randomly select top-left corner within valid region
        top = randint(h_start, h_end - self.HW[0])
        left = randint(w_start, w_end - self.HW[1])

        # Adjust camera intrinsics for crop
        K_crop = K.clone()
        # Shift principal point by crop offset
        K_crop[0] = K[0] * W / self.HW[1]  # rescale to original image
        K_crop[1] = K[1] * H / self.HW[0]  # rescale to original image
        K_crop[0, 2] = K_crop[0, 2] - left  # cx
        K_crop[1, 2] = K_crop[1, 2] - top  # cy

        return (top, left), K_crop

    def forward(self, img, K, params):
        """Apply the random crop to an image using pre-sampled parameters.

        Args:
            img (torch.Tensor): Image tensor with shape [..., C, H, W].
            K (torch.Tensor): Camera intrinsics matrix.
            params (tuple): Parameters for cropping (top, left).

        Returns:
            torch.Tensor: Cropped image of size self.HW.
        """
        top, left = params
        # Perform the crop
        return crop(img, top=top, left=left, height=self.HW[0], width=self.HW[1])


class RandomFocal(Transform):
    """Random focal length augmentation.

    This transform randomly changes the focal length of the camera by resizing and cropping the image.
    The output size is fixed by HW but input size can change. The change is controlled by downsizing
    the full resolution image to a size between original and final size and then cropping to final size.

    Attributes:
        HW (tuple): Target output dimensions (height, width).
    """

    def __init__(self, HW):
        """Initialize the random focal length transform.

        Args:
            HW (tuple): Target output dimensions (height, width).
        """
        super().__init__()
        self.HW = HW

    def sample_params(self, shape, K):
        """Sample focal augmentation parameters and adjust camera intrinsics.

        Args:
            shape (tuple): Input image shape.
            K (torch.Tensor): Camera intrinsics matrix.

        Returns:
            tuple:
                - params (tuple): Parameters containing (scale_factor, top, left, intermediate_size).
                - K_new (torch.Tensor): Adjusted camera intrinsics for the transformed image.
        """
        *_, H, W = shape

        # Calculate scale range based on input and output dimensions
        # Scale range is between target size and input size
        min_scale = max(self.HW[0] / H, self.HW[1] / W)  # Minimum scale to fit target

        # Sample scale factor for focal length simulation
        scale_factor = uniform(min_scale, 1.0)

        # Calculate intermediate size after scaling
        intermediate_H = int(H * scale_factor)
        intermediate_W = int(W * scale_factor)

        # apply rescale
        K[0] = K[0] * intermediate_W / W  # rescale to intermediate image
        K[1] = K[1] * intermediate_H / H  # rescale to intermediate image

        # apply center crop offset
        K[0, 2] = K[0, 2] - (intermediate_W - self.HW[1]) // 2  # cx
        K[1, 2] = K[1, 2] - (intermediate_H - self.HW[0]) // 2  # cy

        return (intermediate_H, intermediate_W), K

    def forward(self, img, params):
        """Apply the random focal augmentation to an image using pre-sampled parameters.

        Args:
            img (torch.Tensor): Image tensor with shape [..., C, H, W].
            K (torch.Tensor): Camera intrinsics matrix.
            params (tuple): Parameters containing (scale_factor, top, left, intermediate_size).

        Returns:
            torch.Tensor: Transformed image of size self.HW.
        """
        H, W = params

        # Resize image to intermediate size
        img_resized = resize(img, size=(H, W))

        # center crop to final size
        img_cropped = center_crop(img_resized, output_size=self.HW)

        return img_cropped
