import numpy as np
import os


def read_calib_file(filepath):
    """Read KITTI calibration file.

    Parses a KITTI calibration file and converts the values to appropriate
    data types (mostly numpy arrays).

    Args:
        filepath (str): Path to the calibration file.

    Returns:
        dict: Dictionary containing the calibration data with keys from the file
            and values converted to appropriate types.
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
    return data


def load_calib_rigid(filepath):
    """Read a rigid transform calibration file as a numpy array.

    Reads a calibration file containing a rigid transformation matrix
    (rotation and translation) and returns the 4x4 transformation matrix.

    Args:
        filepath (str): Path to the calibration file containing 'R' and 'T' entries.

    Returns:
        numpy.ndarray: 4x4 rigid transformation matrix.
    """
    data = read_calib_file(filepath)
    R = data["R"].reshape(3, 3)
    t = data["T"].reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def load_calib_cam_to_cam(path, HW):
    """Load camera-to-camera calibration data from KITTI format.

    Reads camera calibration files and computes intrinsic and extrinsic matrices
    for both left and right cameras, scaled according to the target image dimensions.

    Args:
        path (str): Path to the directory containing calibration files.
        HW (tuple): Target image dimensions as (height, width).

    Returns:
        dict: Dictionary containing calibration data including:
            - K_2, K_3: Intrinsic matrices for left and right cameras.
            - world_to_cam2, world_to_cam3: Transformation matrices from world to camera.
            - baseline: Distance between stereo cameras.
    """
    data = {}
    filedata = read_calib_file(os.path.join(path, "calib_cam_to_cam.txt"))
    # Get rectification matrix (shared between cameras)
    r_rect = np.eye(4)
    r_rect[0:3, 0:3] = np.reshape(filedata["R_rect_00"], (3, 3))

    # Load velodyne transformation
    t_cam0unrect_velo = load_calib_rigid(os.path.join(path, "calib_velo_to_cam.txt"))

    # Get projection matrices for left and right cameras
    for i in [2, 3]:  # 2: left color, 3: right color
        p_rect = np.reshape(filedata[f"P_rect_0{i}"], (3, 4))

        # Scale projection matrix based on new image dimensions
        original_size = filedata[f"S_rect_0{i}"]  # [width, height]
        scale_w = HW[1] / original_size[0]  # new_width / original_width
        scale_h = HW[0] / original_size[1]  # new_height / original_height

        # Scale the focal length and principal point
        p_rect[0, :] *= scale_w
        p_rect[1, :] *= scale_h

        data[f"K_{i}"] = p_rect[0:3, 0:3]

        # Compute velodyne to camera transformations for both cameras
        intrinsic = np.eye(4)
        intrinsic[:3] = p_rect
        data[f"world_to_cam{i}"] = intrinsic @ r_rect @ t_cam0unrect_velo

    # Compute baseline from rectified projection matrices
    # The baseline is encoded in the x-translation (element [0,3]) of the right camera's projection matrix
    # We need to normalize by the focal length to get meters
    left_x = filedata["P_rect_02"][3]  # x-translation for left camera
    right_x = filedata["P_rect_03"][3]  # x-translation for right camera
    focal_length = filedata["P_rect_02"][0]  # focal length (same for both cameras after rectification)
    data["baseline"] = (right_x - left_x) / focal_length

    return data
