#!/usr/bin/env python3
import os
import argparse
import numpy as np

from sob.datasets.load_depth import get_lidar_depth
from sob.datasets.load_calib import load_calib_cam_to_cam  # adjust import if needed


def process_file(list_path, data_path, height, width):
    im_shape = (height, width)
    with open(list_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # skip invalid lines
        folder = parts[0]  # now expected to be full folder string (e.g., "2011_09_26/2011_09_26_drive_0001_sync")
        try:
            frame = int(parts[1])
        except ValueError:
            continue
        side = parts[2]

        # Extract the date part from the folder path for calibration
        sequence_date = folder.split("/")[0]  # e.g., "2011_09_26"
        calib_folder = os.path.join(data_path, sequence_date)

        try:
            calib = load_calib_cam_to_cam(calib_folder, im_shape)
            cam_idx = 2 if side == "l" else 3
            projection_matrix = calib[f"world_to_cam{cam_idx}"]
        except FileNotFoundError:
            print(f"Calibration file not found in {calib_folder}, skipping {folder}/{frame:010d}")
            continue

        # Construct the velodyne file path using folder directly
        velo_filename = os.path.join(data_path, folder, "velodyne_points", "data", "{:010d}.bin".format(frame))
        if not os.path.exists(velo_filename):
            print(f"Velodyne file not found: {velo_filename}")
            continue

        # Compute depth map (shape: (1, H, W))
        depth = get_lidar_depth(velo_filename, projection_matrix, im_shape)
        depth_map = depth[0]  # remove channel dimension if present

        # Create output directory and file name (e.g. data/Kitti/lidar/2011_09_26/2011_09_26_drive_0001_sync/0000000123_l.npy)
        out_dir = os.path.join(data_path, "lidar", folder)
        os.makedirs(out_dir, exist_ok=True)
        output_name = f"{frame:010d}_{side}.npy"
        output_path = os.path.join(out_dir, output_name)
        np.save(output_path, depth_map)
        print(f"Saved depth map to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch export LiDAR depth maps for train and val splits.")
    parser.add_argument("--data_path", type=str, default="data/Kitti", help="Base path to KITTI data")
    parser.add_argument("--height", type=int, default=192, help="Image height after resizing")
    parser.add_argument("--width", type=int, default=640, help="Image width after resizing")
    args = parser.parse_args()

    print("Processing train files...")
    process_file(
        "/home/aloception/external-repos/sob/splits/eigen_zhou/train_files.txt",
        args.data_path,
        args.height,
        args.width,
    )
    print("Processing val files...")
    process_file(
        "/home/aloception/external-repos/sob/splits/eigen_zhou/val_files.txt",
        args.data_path,
        args.height,
        args.width,
    )


if __name__ == "__main__":
    main()
