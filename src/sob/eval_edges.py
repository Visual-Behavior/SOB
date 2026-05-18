import numpy as np

import torch

try:
    import kaolin.metrics.pointcloud as kpm
except ImportError:
    print("Kaolin not found, edge metrics will not be available")

from .utils import get_points_3d
from kornia.morphology import dilation
from kornia.filters import canny
from scipy.interpolate import NearestNDInterpolator
from kornia.morphology import erosion, closing
import torch.nn.functional as F


def map_edges(pred_edges, target_edges, max_dist=10.0):
    """Map edge pixels from prediction to target using KDTree

    Args:
        pred_edges (torch.Tensor): Binary edge mask for predictions shape [H, W]
        target_edges (torch.Tensor): Binary edge mask for target shape [H, W]
        max_dist (float): Maximum distance threshold for valid mappings

    Returns:
        tuple: (pred_coords, target_coords) tensors for valid mappings shape [N,2]
    """
    # Get edge coordinates
    pred_coords = torch.nonzero(pred_edges)
    target_coords = torch.nonzero(target_edges)

    pred_coords_3d = torch.cat([pred_coords, torch.zeros(pred_coords.shape[0], 1, device=pred_edges.device)], dim=1)
    target_coords_3d = torch.cat(
        [target_coords, torch.zeros(target_coords.shape[0], 1, device=target_edges.device)], dim=1
    )

    # Reshape for kaolin (expects [batch_size, num_points, dims])
    pred_points = pred_coords_3d.unsqueeze(0)
    target_points = target_coords_3d.unsqueeze(0)

    # Compute one-sided distances from pred to target
    # import kaolin o
    distances, indices = kpm.sided_distance(pred_points, target_points)

    # Remove batch dimension
    distances = distances.squeeze(0)
    indices = indices.squeeze(0)

    # Filter by distance threshold
    valid_mask = distances < max_dist

    # Get valid mappings
    valid_pred_coords = pred_coords[valid_mask]
    valid_target_indices = indices[valid_mask]
    valid_target_coords = target_coords[valid_target_indices]

    return (valid_pred_coords, valid_target_coords)


@torch.compile
def batch_map_edges(pred_edges, target_edges, max_dist=10.0):
    coords = []
    for i in range(pred_edges.shape[0]):
        coords.append(map_edges_n2(pred_edges[i, 0], target_edges[i, 0], max_dist))
    return coords


def map_edges_n2(pred_edges, target_edges, max_dist=10.0):
    """Optimized version using vectorized operations where possible

    Args:
        pred_edges (torch.Tensor): Binary edge mask for predictions shape [H, W]
        target_edges (torch.Tensor): Binary edge mask for target shape [H, W]
        max_dist (float): Maximum distance threshold for valid mappings

    Returns:
        tuple: (pred_coords, target_coords) tensors for valid mappings shape [N,2]
    """
    # Get edge coordinates
    pred_coords = torch.nonzero(pred_edges)
    target_coords = torch.nonzero(target_edges)

    # Return empty tensors if either is empty
    if pred_coords.shape[0] == 0 or target_coords.shape[0] == 0:
        return (torch.zeros((0, 2), device=pred_edges.device), torch.zeros((0, 2), device=pred_edges.device))

    # Use broadcasting for pairwise distances
    # [N_pred, 1, 2] - [1, N_target, 2] -> [N_pred, N_target, 2]
    diffs = pred_coords.unsqueeze(1) - target_coords.unsqueeze(0)

    # Square distances: [N_pred, N_target]
    distances = torch.sum(diffs**2, dim=2)

    # Find min distance and index for each prediction point
    min_distances, indices = torch.min(distances, dim=1)

    # Filter by distance threshold
    valid_mask = min_distances < max_dist**2  # Using squared distance

    # Get valid mappings
    valid_pred_coords = pred_coords[valid_mask]
    valid_target_indices = indices[valid_mask]
    valid_target_coords = target_coords[valid_target_indices]

    return (valid_pred_coords, valid_target_coords)


def compute_depth_deltas_max_gap(depth, kernel_size=3):
    """Compute depth deltas using maximum gap between center and extrema

    Args:
        depth (torch.Tensor): Depth map, size [B, 1, H, W]
        kernel_size (int): Size of neighborhood

    Returns:
        torch.Tensor: Maximum depth discontinuity shape [B, 1, H, W]
    """

    # Compute max and min values in each neighborhood
    padding = kernel_size // 2
    max_vals = torch.nn.functional.max_pool2d(depth, kernel_size=kernel_size, stride=1, padding=padding)

    min_vals = -torch.nn.functional.max_pool2d(-depth, kernel_size=kernel_size, stride=1, padding=padding)
    return max_vals - min_vals
    # Compute gaps in both directions
    max_center_gap = max_vals - depth
    center_min_gap = depth - min_vals

    # Take the maximum gap
    max_gap = torch.maximum(max_center_gap, center_min_gap)

    # Only keep deltas at edge locations
    return max_gap


def get_edges(depth):
    """Extract edges from depth map using Canny edge detector

    Args:
        depth (torch.Tensor): Depth map tensor, shape [B, 1, H, W]
        mask (torch.Tensor, optional): Optional mask to apply to edges, shape [B, 1, H, W]

    Returns:
        torch.Tensor: Edge mask (binary) shape [B, 1, H, W]
    """
    log_depth = depth.log()
    amin = log_depth.amin(dim=(1, 2, 3), keepdim=True)
    amax = log_depth.amax(dim=(1, 2, 3), keepdim=True)
    scaled_log_depth = (log_depth - amin) / (amax - amin)
    return canny(scaled_log_depth)[1].bool()
    # if mask is not None:
    #     mask = erosion(mask.float(), torch.ones(5, 5, device=mask.device))
    #     return edges[1] * mask


def nearest_neighbor_depth(sparse_depth):
    """
    Resample lidar depth map using nearest neighbor interpolation

    Args:
        depth (torch.Tensor): Depth map tensor

    Returns:
        torch.Tensor: Resampled depth map
    """
    shape = sparse_depth.shape
    device = sparse_depth.device
    sparse_depth = sparse_depth.cpu().numpy().squeeze()
    # Identify valid points (non-zero or non-nan values)
    valid_mask = ~np.isnan(sparse_depth) if np.isnan(sparse_depth).any() else sparse_depth > 0
    y_valid, x_valid = np.where(valid_mask)

    # If no valid points, return the original
    if len(y_valid) == 0:
        return sparse_depth

    # Get values of valid points
    valid_points = np.column_stack([y_valid, x_valid])
    valid_values = sparse_depth[valid_mask]

    # Create interpolator
    interpolator = NearestNDInterpolator(valid_points, valid_values)

    # Create a grid of all coordinates
    h, w = sparse_depth.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Run interpolation
    filled_depth = interpolator(np.column_stack([yy.ravel(), xx.ravel()])).reshape(h, w)

    return torch.tensor(filled_depth, device=device).view(shape)


def batched_masked_median(tensor, mask):
    """Compute median values for each item in batch where mask is True.

    Args:
        tensor (torch.Tensor): Batch of tensors [B, 1, H, W]
        mask (torch.Tensor): Batch of masks [B, 1, H, W]

    Returns:
        torch.Tensor: Median values for each item in batch [B]
    """
    batch_size = tensor.shape[0]
    medians = torch.zeros(batch_size, device=tensor.device)

    for i in range(batch_size):
        valid_values = tensor[i][mask[i]]
        medians[i] = torch.median(valid_values)

    return medians


def geometric_mean(values):
    """Compute geometric mean gap between predicted and ground truth depths.

    Args:
        pred (torch.Tensor): Predicted depth map shape [H, W].
        label (torch.Tensor): Ground truth depth map shape [H, W].

    Returns:
        float: Geometric mean gap.
    """
    log_gap = torch.log(values)
    mean = torch.exp(log_gap.mean())
    # std = torch.exp(log_gap.std())
    return mean  # , std


def gap_distance(gt, pred, edge_gt, edge_pred, mask):
    """Compute matching gap error between predicted and ground truth depths.

    Args:
        gt (torch.Tensor): Ground truth depth map shape [B, 1, H, W].
        pred (torch.Tensor): Predicted depth map shape [B, 1, H, W].
        edge_gt (torch.Tensor): Edge mask shape [B, 1, H, W].
        edge_pred (torch.Tensor): Edge mask shape [B, 1, H, W].
        mask (torch.Tensor): Mask for valid pixels shape [B, 1, H, W].

    Returns:
        tuple: (pred_gaps, gt_gaps) - Raw gap values for valid edge pixels.
            - pred_gaps: Tensor of predicted gap values
            - gt_gaps: Tensor of ground truth gap values
    """
    gap_pred = compute_depth_deltas_max_gap(pred)  # B 1 H W
    gap_gt = compute_depth_deltas_max_gap(gt)  # B 1 H W

    coords = batch_map_edges(edge_pred, edge_gt)

    all_pred_gaps = []
    all_gt_gaps = []

    for i in range(gap_pred.shape[0]):
        pred_coords, gt_coords = coords[i]
        if pred_coords.shape[0] == 0 or gt_coords.shape[0] == 0:
            continue

        # Extract gap values at corresponding coordinates
        pred_gaps = gap_pred[i, 0, pred_coords[:, 0], pred_coords[:, 1]]
        gt_gaps = gap_gt[i, 0, gt_coords[:, 0], gt_coords[:, 1]]

        # Filter valid values
        valid_mask = gt_gaps > 0
        if valid_mask.sum() == 0:
            continue

        # Store raw gap values
        all_pred_gaps.append(pred_gaps[valid_mask])
        all_gt_gaps.append(gt_gaps[valid_mask])

    # Concatenate all pixel-wise values
    if all_pred_gaps and all_gt_gaps:
        all_pred_gaps = torch.cat(all_pred_gaps)
        all_gt_gaps = torch.cat(all_gt_gaps)
    else:
        # Return empty tensors if no valid pixels found
        all_pred_gaps = torch.tensor([], device=gt.device)
        all_gt_gaps = torch.tensor([], device=gt.device)

    return all_pred_gaps, all_gt_gaps


# def get_edges(depth, mask=None):
#     """Extract edges from depth map using Canny edge detector

#     Args:
#         depth (torch.Tensor): Depth map tensor
#         mask (torch.Tensor, optional): Optional mask to apply to edges

#     Returns:
#         torch.Tensor: Edge mask (binary)
#     """
#     log_depth = depth.log()
#     scaled_log_depth = (log_depth - log_depth.min()) / (log_depth.max() - log_depth.min())
#     edges = canny(scaled_log_depth[None, None])
#     if mask is not None:
#         from kornia.morphology import erosion

#         mask = erosion(mask[None, None].float(), torch.ones(5, 5, device=mask.device))
#         return (edges[1] * mask).squeeze()
#     return edges[1].squeeze()


# def get_edges(depth):
#     disp = 1 / depth
#     # get the difference of a depth pixel and it's neighbors
#     diff_r = disp[:, 1:] - disp[:, :-1]  # right neighbor
#     diff_d = disp[1:, :] - disp[:-1, :]  # down neighbor
#     diff_l = disp[:, :-1] - disp[:, 1:]  # left neighbor
#     diff_u = disp[:-1, :] - disp[1:, :]  # up neighbor
#     # pad missing values with 0
#     diff_r = np.pad(diff_r, ((0, 0), (1, 0)), mode="constant", constant_values=0)
#     diff_d = np.pad(diff_d, ((1, 0), (0, 0)), mode="constant", constant_values=0)
#     diff_l = np.pad(diff_l, ((0, 0), (0, 1)), mode="constant", constant_values=0)
#     diff_u = np.pad(diff_u, ((0, 1), (0, 0)), mode="constant", constant_values=0)
#     # get the maximum difference
#     diff = np.max(np.abs(np.stack([diff_r, diff_d, diff_l, diff_u])), axis=0)
#     mask = diff > 0.0075
#     return mask


def scatter_points(loss, points, K, H=375, W=1242):
    image = torch.zeros(H, W, device=points.device)
    points = points
    K = K
    points = points[0] @ torch.tensor(K[:3, :3].T, device=points.device)
    d = points[:, 2:]
    points = points[:, :2] / d
    points = points.round().long()
    mask = (points[:, 1] < image.size(0)) & (points[:, 0] < image.size(1)) & (points[:, 1] > 0) & (points[:, 0] > 0)
    points = points[mask]
    image[points[:, 1], points[:, 0]] = loss[0, mask] / d[mask, 0]
    return image


def pointcloud_error(gt, pred, edge, K_gt):
    """Compute pointcloud error between predicted and ground truth depths.

    Args:
        gt (torch.Tensor): Ground truth depth map shape [H, W].
        pred (torch.Tensor): Predicted depth map shape [H, W].
        edge (torch.Tensor): Edge mask shape [H, W].
        K_gt (torch.Tensor): Camera intrinsic matrix shape [4, 4].

    Returns:
        list: List of metrics including mean distance, and f-scores for different thresholds.
    """
    gt = gt.cuda()
    pred = pred.cuda()
    K_gt = K_gt.cuda()
    edge = edge.cuda()

    scale = torch.median(gt[gt > 0]) / torch.median(pred[gt > 0])
    pred = pred * scale

    gt0 = ((gt > 0) & (gt < 80)).view(-1)
    pred_mask = ((pred > 0) & (pred < 80)).view(-1)

    gt = get_points_3d(gt[None, None], torch.linalg.inv(K_gt)[None]).view(1, 3, -1).permute(0, 2, 1)
    pred = get_points_3d(pred[None, None], torch.linalg.inv(K_gt)[None]).view(1, 3, -1).permute(0, 2, 1)
    h_mask = pred[:, :, 1] > -1.5
    pred_mask = pred_mask * h_mask.view(-1)
    predm = pred[:, pred_mask]

    # sided distance
    d_pred_gt, _ = kpm.sided_distance(predm, gt[:, gt0])
    d_pred_gt = torch.sqrt(d_pred_gt) / predm[:, :, 2]

    sd1 = d_pred_gt.mean().squeeze().cpu().numpy() * 100

    f_scores = [f_score(d_pred_gt, th).squeeze().cpu().numpy() * 100 for th in [0.01, 0.05, 0.1, 0.25]]

    metrics = [[sd1] + f_scores]

    for edge in dilate_edge(edge):
        edge_mask = edge.view(-1)[pred_mask]
        d_pred_edge = d_pred_gt.view(-1)[edge_mask]
        sd1 = d_pred_edge.mean().squeeze().cpu().numpy() * 100
        f_scores = [f_score(d_pred_edge, th).squeeze().cpu().numpy() * 100 for th in [0.01, 0.05, 0.1, 0.25]]
        metrics.append([sd1] + f_scores)

    return metrics


def f_score(points, threshold):
    positives = (points < threshold).sum()
    negatives = points.numel() - positives
    return positives / (positives + 0.5 * negatives)


def dilate_edge(edge):
    # returns dilated edge images with increasing dilation rate
    h, w = edge.shape[-2:]
    dilated_edges = [edge.bool().squeeze()]
    for i in [3, 5, 7]:
        dilated_edges.append(dilation(edge.view(1, 1, h, w), torch.ones(i, i, device=edge.device)).bool().squeeze())
    return dilated_edges


def find_minimum_heights(depth_map):
    """
    For each row in a sparse LiDAR depth map, find the minimum height starting from
    which there is a point.

    Args:
        depth_map: Tensor of shape (B, 1, H, W) representing sparse LiDAR depth maps
                  Non-zero values indicate points.

    Returns:
        Tensor of shape (B, W) containing the minimum height for each column where a point exists.
        If a column has no points, the value will be H.
    """
    B, _, H, W = depth_map.shape

    # Create a boolean mask where True indicates presence of a point
    mask = depth_map > 0

    # Find the indices of all points (reshape to B x W x H for easier processing)
    mask = mask.squeeze(1).permute(0, 2, 1)  # B x W x H

    # For each batch and column, find the minimum height where a point exists
    # First create a height tensor that repeats for all positions
    height_indices = torch.arange(H, device=depth_map.device).expand(B, W, H)

    # Apply mask to get only heights where points exist
    masked_heights = height_indices.masked_fill(~mask, H)

    # Get minimum height for each column
    min_heights = masked_heights.min(dim=2)[0]  # B x W

    return min_heights


def find_minimum_heights_mask(depth_map):
    """
    For each row in a sparse LiDAR depth map, find the minimum height starting from
    which there is a point and return a mask with 1s for all points under that height.

    Args:
        depth_map: Tensor of shape (B, 1, H, W) representing sparse LiDAR depth maps
                  Non-zero values indicate points.

    Returns:
        Tuple containing:
        - Tensor of shape (B, W) containing the minimum height for each column where a point exists.
          If a column has no points, the value will be H.
        - Tensor of shape (B, 1, H, W) with 1s for all points under the minimum height found.
    """
    B, _, H, W = depth_map.shape

    # Get minimum heights for each column
    min_heights = find_minimum_heights(depth_map)  # B x W

    # Create height indices tensor of shape (H)
    height_indices = torch.arange(H, device=depth_map.device)

    # Create a mask of shape (B, W, H) where True means the height is less than the min height
    # First, expand min_heights to (B, W, 1)
    min_heights_expanded = min_heights.unsqueeze(-1)  # B x W x 1

    # Then expand height indices to (1, 1, H)
    height_indices_expanded = height_indices.view(1, 1, H)  # 1 x 1 x H

    # Compare heights with min heights (broadcasting happens automatically)
    # True where height < min_height
    under_min_mask = height_indices_expanded > min_heights_expanded  # B x W x H

    # Reshape to match the input format (B, 1, H, W)
    under_min_mask = under_min_mask.permute(0, 2, 1).unsqueeze(1)  # B x 1 x H x W

    # Convert boolean mask to binary (0, 1) mask
    binary_mask = under_min_mask.float()

    # close the mask
    binary_mask = closing(binary_mask, torch.ones(3, 3, device=binary_mask.device))

    return binary_mask


def patch_bernoulli_entropy(image, K=3):
    """
    For each patch of size k x k in the input image with given stride (default is non-overlapping),
    compute:
      - the minimum and maximum values of the patch,
      - normalize the patch to [0,1],
      - compute the Bernoulli entropy for each value:
            H(p) = -p*log2(p) - (1-p)*log2(1-p),
      - and average the entropy over the patch.

    Args:
        image (torch.Tensor): Input image tensor, shape [B, 1, H, W].
        k (int): Patch size (both height and width).
        stride (int, optional): Stride for patch extraction. If None, defaults to k (non-overlapping).

    Returns:
        torch.Tensor: Tensor of patch-averaged entropies with shape [B, out_h, out_w],
                      where out_h = 1 + (H - k) // stride and out_w = 1 + (W - k) // stride.
    """
    B, C, H, W = image.shape

    # pad the image to handle edge cases
    pad = K // 2
    image = F.pad(image, (pad, pad, pad, pad), mode="replicate")

    # Extract patches using unfold
    patches = F.unfold(image, K)  # [B, CKK, HW]

    # Compute the min and max for each patch along the patch dimension.
    min_vals, max_vals = torch.aminmax(patches, dim=1, keepdim=True)

    # Normalize patches to [0, 1], adding a small epsilon for numerical stability.
    p = (patches - min_vals) / (max_vals - min_vals + 1e-7)

    # Compute Bernoulli entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
    entropy = -(p * torch.log2(p + 1e-7) + (1 - p) * torch.log2(1 - p + 1e-7))

    # Average the entropy over the patch dimension and reshape to input size
    return entropy.mean(dim=1).view(B, 1, H, W)


def edge_entropy(depth):
    """
    Compute the entropy of edges in a depth map.

    Args:
        depth (torch.Tensor): Depth map tensor, shape [B, 1, H, W].
        mask (torch.Tensor): Mask tensor, shape [B, 1, H, W].

    Returns:
        torch.Tensor: Tensor of edge entropies with shape N.
    """
    edges = get_edges(depth)

    entropy = patch_bernoulli_entropy(depth, 3)

    return entropy[edges]
