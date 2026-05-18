import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np


class VKitti(Dataset):
    """Dataset class for Virtual KITTI (VKitti) data.

    This dataset is designed for evaluation purposes in self-supervised depth
    estimation. It loads RGB images and ground truth depth maps from the VKitti
    dataset, which contains synthetic driving scenes modeled after real KITTI data.

    The dataset structure is expected to be:
    VKitti/
        Scene01/
            clone/
                frames/
                    rgb/
                        Camera_0/
                            rgb_00000.jpg
                            ...
                        Camera_1/
                            rgb_00000.jpg
                            ...
                    depth/
                        Camera_0/
                            depth_00000.png
                            ...
                        Camera_1/
                            depth_00000.png
                            ...

    Attributes:
        scenes (list): List of scene names and their frame counts.
        intrinsics (torch.Tensor): Camera intrinsic matrix for VKitti.
    """

    scenes = [
        ("Scene01", 447),
        ("Scene02", 223),
        ("Scene06", 270),
        ("Scene18", 339),
        ("Scene20", 837),
    ]
    intrinsics = torch.tensor([[725.0087, 0, 620.5], [0, 725.0087, 187], [0, 0, 1]])

    def __init__(self, root, HW=(192, 640), sources=(-1, 1)):
        """Initialize the VKitti dataset.

        Args:
            root (str): Path to the VKitti dataset root directory.
            HW (tuple, optional): Target image dimensions as (height, width).
                Defaults to (192, 640).
            sources (tuple, optional): Temporal sources to use as frame offsets.
                For example, (-1, 1) for previous and next frames or 0 for stereo.
                Can also use larger offsets like (-2, -1, 1, 2).
                Defaults to (-1, 1).
        """
        self.root = root
        self.H = HW[0]
        self.W = HW[1]
        self.sources = sources
        self.cumulative_frames = [0]
        for _, num_frames in self.scenes:
            self.cumulative_frames.append(self.cumulative_frames[-1] + num_frames * 2)

    def __len__(self):
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of samples, which is the sum of frames in all scenes,
                multiplied by 2 to account for stereo views.
        """
        return sum(num_frames for _, num_frames in self.scenes) * 2  # 2 because of stereo

    def __getitem__(self, idx):
        """Get a sample from the dataset at the given index.

        This method:
        1. Determines the scene, frame, and camera indices
        2. Loads the RGB image and depth map
        3. Prepares the camera intrinsics
        4. Loads source frames for self-supervised learning

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing:
                - 'target': Target RGB image tensor.
                - 'K': Camera intrinsics tensor.
                - 'depth_gt': Ground truth depth map tensor.
                - 'source_{name}': Source RGB image tensors for each source.
        """
        # Determine the scene index using cumulative frames
        scene_idx = next(i for i, total in enumerate(self.cumulative_frames) if idx < total) - 1
        frame_idx = (idx - self.cumulative_frames[scene_idx]) // 2
        camera_idx = (idx - self.cumulative_frames[scene_idx]) % 2

        scene_name, _ = self.scenes[scene_idx]

        # Construct paths for image, depth, and intrinsic
        rgb_path = os.path.join(
            self.root, scene_name, "clone", "frames", "rgb", f"Camera_{camera_idx}", f"rgb_{frame_idx:05d}.jpg"
        )
        depth_path = os.path.join(
            self.root, scene_name, "clone", "frames", "depth", f"Camera_{camera_idx}", f"depth_{frame_idx:05d}.png"
        )

        # Load image and depth
        image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.W, self.H))
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        intrinsics = self.intrinsics.clone()
        intrinsics[:2] = intrinsics[:2] * 0.5

        # Load target image and create base inputs dict
        inputs = {"target": torch.tensor(image).permute(2, 0, 1) / 255, "K": intrinsics, "index": idx}

        # Load source images
        for source in self.sources:
            source_frame_idx = frame_idx + source
            # Skip if frame would be out of bounds
            if 0 <= source_frame_idx < self.scenes[scene_idx][1]:
                source_path = os.path.join(
                    self.root,
                    scene_name,
                    "clone",
                    "frames",
                    "rgb",
                    f"Camera_{camera_idx}",
                    f"rgb_{source_frame_idx:05d}.jpg",
                )
                source_image = cv2.cvtColor(cv2.imread(source_path), cv2.COLOR_BGR2RGB)
                source_image = cv2.resize(source_image, (self.W, self.H))
                inputs[f"source_{source}"] = torch.tensor(source_image).permute(2, 0, 1) / 255

        return inputs, torch.tensor(depth.astype(np.float32)).unsqueeze(0) * 0.01
