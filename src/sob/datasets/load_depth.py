import numpy as np


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(" "))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def get_lidar_depth(velo_filename, projection_matrix, im_shape):
    velo = load_velodyne_points(velo_filename)
    return generate_depth_map(velo, projection_matrix, im_shape)


def generate_depth_map(velo, projection_matrix, im_shape):
    """Generate a depth map from velodyne data.
    Args:
        velo_filename: path to velodyne points
        projection_matrix: pre-computed projection matrix
        im_shape: tuple of (height, width)
    """
    # load velodyne points and remove all behind image plane (approximation)
    velo = velo[velo[:, 0] > 0, :]

    # project the points to the camera
    velo_pts_im = velo.dot(projection_matrix.T)  # projects each 4D point with shape (N,4)
    velo_pts_im[:, :2] /= velo_pts_im[:, 2:3]  # element-wise division using broadcasting

    # check if in bounds
    velo_pts_im[:, :2] = np.round(velo_pts_im[:, :2]) - 1
    val_inds = (velo_pts_im[:, :3] >= 0).all(1) & (velo_pts_im[:, :2] < [im_shape[1], im_shape[0]]).all(1)
    velo_pts_im = velo_pts_im[val_inds, :]

    # compute a flattened index for each 2D pixel location in the image
    rows = velo_pts_im[:, 1].astype(np.int32)
    cols = velo_pts_im[:, 0].astype(np.int32)
    inds = np.ravel_multi_index((rows, cols), dims=im_shape)

    # create a depth map (flattened) filled with np.inf, then compute the minimum depth at each index
    depth_flat = np.full(np.prod(im_shape), np.inf, dtype=velo_pts_im.dtype)
    np.minimum.at(depth_flat, inds, velo_pts_im[:, 2])

    # reshape and clean up the depth map
    depth = depth_flat.reshape(im_shape)
    depth[np.isinf(depth)] = 0  # assign 0 to pixels not hit

    return depth[None]
