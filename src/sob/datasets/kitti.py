import os
import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import decode_image
from kornia.augmentation import ColorJitter, RandomHorizontalFlip
from ..utils import readlines
from .load_calib import load_calib_cam_to_cam


class KittiDataset(Dataset):
    """KITTI Dataset for self-supervised depth and pose estimation.

    This dataset provides synchronized stereo and monocular image sequences from the
    KITTI dataset, along with camera calibration information. It supports various
    training modes including temporal (prev/next frames) and stereo supervision.

    Attributes:
        side_map (dict): Mapping from side name to camera ID.
        opposite_side (dict): Mapping from side to opposite side.
        baseline_sign (dict): Sign of the baseline for each side.
        sequences (list): List of valid KITTI date sequences.
        data_path (str): Path to the KITTI dataset.
        H (int): Image height after resizing.
        W (int): Image width after resizing.
        HW (tuple): (height, width) tuple.
        sources (tuple): Source frame for self-supervision.
        img_ext (str): Image file extension.
        split (str): Dataset split ('train', 'val', 'eigen', etc.).
        filenames (list): List of filenames for the dataset.
    """

    side_map = {"l": 2, "r": 3}
    opposite_side = {"l": "r", "r": "l"}
    baseline_sign = {"l": -1, "r": 1}
    sequences = ["2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]

    def __init__(
        self,
        data_path="data",
        height=192,
        width=640,
        sources=(-1, 1),
        img_ext=".jpg",
        split=None,
    ):
        """Initialize the KITTI dataset.

        Args:
            data_path (str, optional): Path to the data directory. Defaults to "data".
            height (int, optional): Height to resize images to. Defaults to 192.
            width (int, optional): Width to resize images to. Defaults to 640.
            sources (tuple, optional): Source frame types for self-supervision.
                Can include negative and positive integers representing frame offsets, 0 stereo.
                For example: (-1, 1) for previous and next frames.
                Can also use larger offsets like (-2, -1, 1, 2) for additional frames.
                Defaults to (-1, 1).
            img_ext (str, optional): Image file extension. Defaults to ".jpg".
            split (str, optional): Dataset split ('train', 'val', 'eigen', etc.).
                Defaults to None.
        """
        self.data_path = os.path.join(data_path, "Kitti")
        if split == "eigen":
            filenames_path = os.path.join(self.data_path, "splits/eigen/test_files.txt")
        elif split == "eigen_benchmark":  # KEB
            filenames_path = os.path.join(self.data_path, "splits/eigen_benchmark/test_files.txt")
        else:
            if split == "val":
                split_name = "val_files_sorted"
            elif split == "test":  # KEZ
                split_name = "test_files"
            elif split == "train":
                split_name = "train_files"
            else:
                raise ValueError(f"Unknown split: {split}")
            filenames_path = os.path.join(self.data_path, f"splits/eigen_zhou/{split_name}.txt")

        self.H = height
        self.W = width
        self.HW = (self.H, self.W)
        self.sources = tuple(sources)
        self.img_ext = img_ext
        self.split = split
        self.filenames = readlines(filenames_path)
        self.resize = v2.Resize(self.HW)
        self.colorjitter = ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, same_on_batch=True, p=0.5
        )
        self.flip = RandomHorizontalFlip(p=0.5, same_on_batch=True)
        self.to_float32 = v2.ToDtype(torch.float32, scale=True)
        self.intrinsics, self.projection_matrices, self.baselines = self.load_calibrations()

    def __len__(self):
        """Get the number of samples in the dataset.

        Returns:ColorJitter
            int: Number of samples in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, index):
        """Get an item from the dataset.

        This method loads and prepares all the data for a single sample, including
        target and source images, camera parameters.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary containing:
                - "K": intrinsic matrix (3, 3)
                - "T": extrinsic transformation matrix (4, 4)
                - "baseline": baseline between stereo pairs
                - "target": target image (3, H, W)
                - "source_<source>": source image(s) (3, H, W)
        """
        line = self.filenames[index].split()
        folder = line[0]
        sequence = folder.split("/")[0]
        frame_index = int(line[1])
        side = line[2]
        cam_idx = self.side_map[side]

        inputs = {
            "K": torch.tensor(self.intrinsics[sequence][cam_idx], dtype=torch.float32),  # (3, 3)
            "T": torch.tensor(self.projection_matrices[sequence][cam_idx], dtype=torch.float32),  # (4, 4)
            # "baseline": torch.tensor(self.baselines[sequence] * self.baseline_sign[side], dtype=torch.float32),
            "baseline": torch.tensor(0.02 * self.baseline_sign[side], dtype=torch.float32),
        }
        frame_ids = [0] + list(self.sources)
        frame_names = ["target"] + [f"source_{id}" for id in self.sources]
        sides = [side] + [self.opposite_side[side] if frame_id == 0 else side for frame_id in self.sources]
        offsets = [0 if frame_id == 0 else frame_id for frame_id in frame_ids]

        images = [self.get_color(folder, frame_index + offset, side) for side, offset in zip(sides, offsets)]
        images = self.resize(torch.stack(images)) / 255.0  # (B, 3, H, W)

        if self.split == "eigen_benchmark" or self.split == "eigen" or self.split == "test":
            inputs.update({f"{name}": img for name, img in zip(frame_names, images)})
            return inputs

        depth = None
        if self.split == "train":
            images = self.flip(images)
            # depth = self.flip(depth[None], params=self.flip._params)[0]
            if self.flip._params["batch_prob"].any():
                inputs = self.flip_projections(inputs)

            images_aug = self.colorjitter(images)
            inputs.update({f"{name}_aug": img for name, img in zip(frame_names, images_aug)})
        else:
            depth = self.get_depth(folder, frame_index, side)

        inputs.update({f"{name}": img for name, img in zip(frame_names, images)})

        if depth is None:
            return inputs

        return inputs, depth

    def get_color(self, folder, frame_index, side):
        """Load color image from the dataset.

        Args:
            folder (str): Path to the folder containing the sequence.
            frame_index (int): Index of the frame to load.
            side (str): Camera side ('l' for left, 'r' for right).

        Returns:
            torch.Tensor: Color image tensor.
        """
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return decode_image(image_path)

    def get_depth(self, folder, frame_index, side):
        """Load depth map from preprocessed files.

        Args:
            folder (str): Path to the folder containing the sequence.
            frame_index (int): Index of the frame to load.
            side (str): Camera side ('l' for left, 'r' for right).

        Returns:
            torch.Tensor: Depth map tensor.
        """
        # Load from preprocessed depth map file
        depth_filename = os.path.join(self.data_path, "lidar", folder, f"{frame_index:010d}_{side}.npy")
        return torch.from_numpy(np.load(depth_filename)[None]).float()

    def flip_projections(self, inputs):
        """Adjust camera intrinsics and extrinsics after horizontal flip.

        Args:
            inputs (dict): Dictionary containing camera parameters that need adjustment.
            is_flipped (bool): Whether images have been flipped.
        """
        K4 = torch.eye(4, dtype=torch.float32)
        K4[:3, :3] = inputs["K"]
        T = torch.inverse(K4) @ inputs["T"]
        inputs["K"][0, 2] = self.W - inputs["K"][0, 2]
        K4[:3, :3] = inputs["K"]
        inputs["T"] = K4 @ T
        inputs["baseline"] = -inputs["baseline"]
        return inputs

    def load_calibrations(self):
        """Load all calibration data for the dataset.

        Loads camera intrinsics, projection matrices, and stereo baselines for
        all sequences in the dataset.

        Returns:
            tuple:
                - intrinsics (dict): Intrinsic matrices for all cameras and sequences.
                - projection_matrices (dict): Projection matrices for all cameras and sequences.
                - baselines (dict): Stereo baselines for all sequences.
        """
        intrinsics = {}
        projection_matrices = {}
        baselines = {}
        for seq in self.sequences:
            intrinsics[seq] = {}
            projection_matrices[seq] = {}
            calib = load_calib_cam_to_cam(os.path.join(self.data_path, seq), self.HW)

            for cam_idx in [2, 3]:  # 2: left, 3: right
                intrinsics[seq][cam_idx] = calib[f"K_{cam_idx}"]
                projection_matrices[seq][cam_idx] = calib[f"world_to_cam{cam_idx}"]
                baselines[seq] = calib["baseline"]

        return intrinsics, projection_matrices, baselines
