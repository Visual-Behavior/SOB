import torch
import numpy as np
from torch.utils.data import DataLoader
from sob.utils import track

from kornia.filters import gaussian_blur2d
from sob.datasets.kitti import KittiDataset
from sob.eval_edges import (
    nearest_neighbor_depth,
    get_edges,
    gap_distance,
    find_minimum_heights_mask,
)
import torch.nn.functional as F


def process_batch(target_batch, device):
    """Process a batch of samples to evaluate edge accuracy

    Args:
        inputs_batch (dict): Dictionary of input tensors
        target_batch (tensor): Target depth tensors
        trainer (Trainer): Model trainer instance
        device (torch.device): Device to run computations on
        use_mode (bool): Whether to use depth_mode_alpha in addition to depth_mean

    Returns:
        tuple: Raw predicted and ground truth gap values for edge pixels.
              For mean prediction and optionally mode prediction:
              (pred_gaps, gt_gaps)
    """
    with torch.no_grad():
        # Move input batch to device
        target_batch = target_batch.to(device)

        # Get individual items
        target = target_batch
        mask_100 = (target < 100) & (target > 0)
        mask_100 &= find_minimum_heights_mask(target).bool()

        # Get edges from ground truth with sky mask
        edges = get_edges(target, mask_100)  # B 1 H W
        pred_depth_mean = target  # gaussian_blur2d(target, 3, (1, 1))
        edge_pred_mean = get_edges(pred_depth_mean, mask_100)  # B 1 H W
        mean_raw_values = gap_distance(target, pred_depth_mean, edges, edge_pred_mean, mask_100)

        return mean_raw_values


def compute_metrics(pred_gaps, gt_gaps):
    """Compute various metrics from raw gap values.

    Args:
        pred_gaps (torch.Tensor): Predicted gap values
        gt_gaps (torch.Tensor): Ground truth gap values

    Returns:
        dict: Dictionary of metrics
    """
    # Compute relative ratio (pred/gt)
    rel_ratio = pred_gaps / gt_gaps

    # Compute absolute relative error (|pred-gt|/gt)
    abs_rel_error = torch.abs(pred_gaps - gt_gaps) / gt_gaps

    # Calculate metrics
    metrics = {
        "rel_ratio_geo_mean": torch.exp(torch.mean(torch.log(rel_ratio))).item(),
        "abs_rel_error_mean": abs_rel_error.mean().item(),
        "abs_rel_error_median": abs_rel_error.median().item(),
        "rel_ratio_mean": rel_ratio.mean().item(),
        "rel_ratio_median": rel_ratio.median().item(),
    }

    return metrics


def main():

    # Set CUDA device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = KittiDataset(split="test")

    # Set up DataLoader with multiple workers
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Initialize tensors to collect raw values
    all_mean_pred_gaps = []
    all_mean_gt_gaps = []

    # Process batches using DataLoader
    for inputs_batch, target_batch in track(dataloader):
        target_batch = F.max_pool2d(torch.stack([nearest_neighbor_depth(t) for t in target_batch]), 5, 1, 2)

        batch_mean_raw = process_batch(target_batch, device)

        # Collect mean raw values
        pred_gaps_mean, gt_gaps_mean = batch_mean_raw
        if pred_gaps_mean.numel() > 0:
            all_mean_pred_gaps.append(pred_gaps_mean.cpu())
            all_mean_gt_gaps.append(gt_gaps_mean.cpu())

    # Concatenate all collected raw values
    all_mean_pred_gaps = torch.cat(all_mean_pred_gaps)
    all_mean_gt_gaps = torch.cat(all_mean_gt_gaps)

    # Compute metrics for mean prediction
    mean_metrics = compute_metrics(all_mean_pred_gaps, all_mean_gt_gaps)

    # Print results
    print("\nEvaluation Results:")
    print("\nMean prediction metrics:")
    for metric_name, value in mean_metrics.items():
        print(f"  {metric_name}: {value:.5f}")


if __name__ == "__main__":
    main()
