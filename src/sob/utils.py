import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    Text,
    ProgressColumn,
)
from typing import Sized, Union


def readlines(filename):
    """Read all the lines in a text file and return as a list.

    Args:
        filename (str): Path to the text file to be read.

    Returns:
        list: A list of strings where each string is a line from the file.
    """
    with open(filename, "r") as f:
        lines = f.read().splitlines()
    return lines


def get_transform_matrix(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix.

    Args:
        axisangle (torch.Tensor): Axis-angle representation of rotation.
        translation (torch.Tensor): Translation vector.
        invert (bool, optional): Whether to invert the transformation. Defaults to False.

    Returns:
        torch.Tensor: 4x4 transformation matrix.
    """
    R = rot_from_axisangle(axisangle.unsqueeze(1))
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix.

    Args:
        translation_vector (torch.Tensor): 3D translation vector.

    Returns:
        torch.Tensor: 4x4 transformation matrix with the translation component.
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix.

    Args:
        vec (torch.Tensor): Axis-angle representation, shape Bx1x3.

    Returns:
        torch.Tensor: 4x4 rotation matrix.

    Note:
        Adapted from https://github.com/Wallacoloo/printipi
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def inv(x):
    """Compute the inverse (1/x) of a tensor or value.

    Args:
        x (torch.Tensor or float): Input value to invert.

    Returns:
        torch.Tensor or float: Inverted value (1/x).
    """
    return 1 / x


def K3_to_K4(K3):
    """Convert a 3x3 camera intrinsic matrix to a 4x4 matrix.

    Args:
        K3 (torch.Tensor): ...x3x3 camera intrinsic matrix.

    Returns:
        torch.Tensor: ...x4x4 camera intrinsic matrix.
    """
    batch_dims = K3.shape[:-2]
    K4 = torch.zeros((*batch_dims, 4, 4), device=K3.device, dtype=K3.dtype)
    K4[..., :3, :3] = K3
    K4[..., 3, 3] = 1
    return K4


def get_points_3d(depth, cam2world):
    """Convert a depth map to 3D points.
    Args:
        depth (torch.Tensor): Depth map shape [B, 1, H, W].
        cam2world (torch.Tensor): Camera to world transformation matrix shape [B, 4, 4].

    Returns:
        torch.Tensor: 3D points shape [B, H, W, 3].
    """
    b, _, h, w = depth.shape

    H = torch.linspace(0, h - 1, h, dtype=torch.float, device=depth.device).view(1, -1, 1).expand(b, -1, w)
    W = torch.linspace(0, w - 1, w, dtype=torch.float, device=depth.device).view(1, 1, -1).expand(b, h, -1)
    D = depth[:, 0]
    points = torch.bmm(
        torch.stack((W * D, H * D, D, torch.ones_like(W)), dim=-1).view(b, -1, 4),
        cam2world.transpose(-1, -2).to(depth.device),
    )
    return (points[..., :3] / points[..., 3:4]).view(b, h, w, 3).permute(0, 3, 1, 2)


class IterSpeed(ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        data_speed = int(speed)
        return Text(f"{data_speed}it/s", style="progress.data.speed")


def track(sequence: Sized):
    """
    A custom tracking function that uses our defined progress_bar.

    Args:
        sequence: An iterable to track progress through
        description: Text description for the progress bar

    Returns:
        Yields items from the sequence
    """

    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        IterSpeed(),
    )

    with progress_bar:
        # Get the total length of the sequence
        total = len(sequence)
        # Create a task
        task_id = progress_bar.add_task("", total=total)

        # Yield items with progress updates
        for item in sequence:
            yield item
            progress_bar.update(task_id, advance=1)


class Table:
    def __init__(self, data: Union[dict, list, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            if isinstance(data, dict):
                data = [data]
            self.df = pd.DataFrame(data)

    def __repr__(self):
        return self.df.to_markdown(index=False)

    def write(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.df.to_markdown(index=False))
        print(f"Metrics saved in {filepath}")

    def write_latex(self, filepath: str):
        with open(filepath, "w") as f:
            f.write(self.df.to_latex(index=False))

    @staticmethod
    def read_table(filepath: str) -> pd.DataFrame:
        """
        Read a markdown table from a file and return it as a pandas DataFrame.
        """
        df = pd.read_table(filepath, sep="|", header=0).dropna(axis=1, how="all").iloc[1:]
        df.columns = [col.strip() for col in df.columns if col.strip()]

        return df

    @classmethod
    def from_file(cls, filepath: str) -> "Table":
        return cls(cls.read_table(filepath))

    @classmethod
    def combine_files(cls, filepaths: list) -> "Table":
        """
        Combine Table instances from a list of markdown files into one Table.
        Adds a new column 'run' (as the first column) extracted from the folder name.
        """
        import os

        dfs = []
        for filepath in filepaths:
            try:
                df = cls.read_table(filepath)
                # Extract run_name from the parent directory of the file
                run_name = os.path.basename(os.path.dirname(filepath))
                # Insert the 'run' column at the beginning (index 0)
                df.insert(0, "run", run_name)
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        combined_df = pd.concat(dfs, ignore_index=True)
        return cls(data=combined_df)
