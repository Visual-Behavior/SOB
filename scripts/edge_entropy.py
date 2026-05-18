import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sob.utils import track, Table

from sob import TrainingConfig, Trainer
from sob.datasets.kitti import KittiDataset
from sob.eval_edges import get_edges, gap_distance, find_minimum_heights_mask, edge_entropy
import torch.nn.functional as F


def process_batch(inputs_batch, trainer, device, use_mean=False):
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
        target_images = inputs_batch["target"].to(device)

        # Get predictions for the whole batch
        preds = trainer.inference_step(target_images)

        # # Get individual items
        # target = target_batch
        # mask = (target < 100) & (target > 0)
        # mask &= find_minimum_heights_mask(target).bool()

        mode_edge_entropy = edge_entropy(preds.depth_mode_alpha())

        if use_mean:
            mean_edge_entropy = edge_entropy(preds.depth_mean())
        else:
            mean_edge_entropy = None

        return mean_edge_entropy, mode_edge_entropy


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate edge prediction accuracy")
    parser.add_argument("--run_name", type=str, default="", help="Model name")
    parser.add_argument("--use_mean", action="store_true", help="Use depth_mode_alpha in addition to depth_mean")
    parser.add_argument("--KEZ", action="store_true", help="Use KEZ split for evaluation.")
    parser.add_argument(
        "--project_path",
        type=str,
        help="Project path to load the model from.",
        default="/home/aloception/.aloception/sob/",
    )
    args = parser.parse_args()

    # Set CUDA device if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model from a saved checkpoint
    model_name = args.run_name
    config = TrainingConfig.from_run(args.run_name, args.project_path)
    config.project_path = args.project_path
    config.data_path = "data"
    config.no_compile = True
    trainer = Trainer(config)

    dataset = KittiDataset(split="test", sources=tuple())

    # Set up DataLoader with multiple workers
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Initialize tensors to collect raw values
    mode_edge_entropy = []
    if args.use_mean:
        mean_edge_entropy = []

    # Process batches using DataLoader
    for inputs_batch in track(dataloader):
        # Process the batch
        mean_ee, mode_ee = process_batch(inputs_batch, trainer, device, use_mean=args.use_mean)
        mode_edge_entropy.append(mode_ee)
        if args.use_mean:
            mean_edge_entropy.append(mean_ee)

    mode_edge_mean = torch.cat(mode_edge_entropy).mean()

    table = Table({"Edge Entropy": mode_edge_mean})
    split = "KEZ" if args.KEZ else "KEB"
    table.write(os.path.join(config.project_path, config.run_name, f"edge_entropy{split}.md"))

    # Print results
    print("Model:", model_name)
    print("\nMode prediction metrics:")
    print(table)

    if args.use_mean:
        print("\nMean prediction metrics:")
        print(f"  Edge Entropy: {torch.cat(mean_edge_entropy).mean():.5f}")


if __name__ == "__main__":
    main()
