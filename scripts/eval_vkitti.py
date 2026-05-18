import os
import numpy as np
import torch
from argparse import ArgumentParser
from sob import Trainer, TrainingConfig
from sob.datasets.vkitti import VKitti
from sob.evaluator import Table
from sob.utils import track
from torch.utils.data import DataLoader
from sob.metrics import metrics_benchmark
from sob.evaluator import Evaluator
from sob.eval_edges import get_edges


def get_predictions(config, trainer):
    dataset = VKitti("data/VKitti2", sources=tuple())  # No source frames needed for evaluation
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False
    )

    depth_predictions = []
    targets = []
    Ks = []
    edges = []

    for batch, gt_depth in track(dataloader):
        gt_depth = gt_depth.cuda()
        K = batch["K"].cuda()
        distribution = trainer.inference_step(batch["target"].cuda())
        # Get depth predictions from the model
        pred_depth = distribution.disp_mode_alpha()[:, 0]
        depth_predictions.append(pred_depth.cpu())
        Ks.append(batch["K"])
        edges.append(get_edges(gt_depth).cpu())

        mask = filter_height(gt_depth, K, h=0.5)
        gt_depth[~mask] = 0
        targets.append(gt_depth.cpu())

    return (
        torch.cat(depth_predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(Ks, dim=0),
        torch.cat(edges, dim=0),
    )


def filter_height(depth, K, h=0.5):
    """
    Filter out pixels with height below h meters above the camera.
    depth: B x 1 x H x W or B x H x W
    K: B x 3 x 3
    h: height in meters above the camera to filter out
    """
    # Remove channel dimension if present
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    # depth: B x H x W
    B, H, W = depth.shape
    device = depth.device

    # Get the camera intrinsics
    fy = K[:, 1, 1].view(B, 1, 1)  # B x 1 x 1
    cy = K[:, 1, 2].view(B, 1, 1)  # B x 1 x 1

    # Compute the height (y-coordinate in camera frame) for each pixel
    # y = (v - cy) * depth / fy
    v_coords = torch.arange(H, device=device).view(1, H, 1)  # 1 x H x 1
    v_grid = v_coords.expand(B, H, W)  # B x H x W
    h_map = -(v_grid - cy) * depth / fy  # B x H x W

    # Filter out pixels with height less than h (above ground)
    mask = h_map < h  # B x H x W

    return mask.unsqueeze(1)  # B x 1 x H x W


def main():
    parser = ArgumentParser(description="Script to evaluate network predictions on VKITTI dataset.")
    parser.add_argument("--run_name", type=str, help="Run name to load", required=True)
    parser.add_argument("--best", action="store_true", help="Use best checkpoint for evaluation.")
    parser.add_argument(
        "--project_path",
        type=str,
        help="Project path to load the model from.",
        default="/home/aloception/.aloception/sob/",
    )
    args = parser.parse_args()

    config = TrainingConfig.from_run(args.run_name, args.project_path)
    config.project_path = args.project_path
    config.data_path = "data"
    config.no_compile = True
    config.load_best = args.best
    trainer = Trainer(config)

    print(f"\n-> Computing predictions on VKITTI")
    preds, targets, K, edges = get_predictions(config, trainer)

    print(f"\n-> Computing metrics...")
    evaluator = Evaluator(mode="mono", metrics=["ibims"])

    targets = {"depth": targets.squeeze(), "K": K, "edge": edges.squeeze()}

    metrics = evaluator.run(preds, targets)

    cp = "best" if args.best else "last"
    metrics_path = os.path.join(config.project_path, config.run_name, f"vkitti_metrics_{cp}.md")
    metrics.write(metrics_path)
    print(metrics)


if __name__ == "__main__":
    main()
