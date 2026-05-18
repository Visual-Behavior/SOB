"""Script to evaluate network predictions on a target dataset."""

import os
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from sob import Trainer, TrainingConfig
from sob.evaluator import get_predictions, Evaluator


def main():
    parser = ArgumentParser(description="Script to evaluate network predictions on a target dataset.")
    parser.add_argument("--run_name", default="weights", type=str, help="Run name to load")
    parser.add_argument("--pc_eval", action="store_true", help="Pointcloud evaluation, takes longer.")
    parser.add_argument(
        "--project_path",
        type=str,
        help="Project path to load the model from.",
        default="",
    )
    args = parser.parse_args()

    config = TrainingConfig.from_run(args.run_name, args.project_path)
    config.project_path = args.project_path
    config.no_compile = True
    trainer = Trainer(config)

    metrics = ["benchmark", "likelihood"]
    if args.pc_eval:
        metrics.append("pointcloud")

    evaluator = Evaluator("mono", metrics=metrics)

    print(f"\n-> Loading targets")
    split = "eigen_benchmark"
    labels = np.load(f"splits/{split}/targets_test.npz", allow_pickle=True)

    print(f"-> Computing predictions on {split}")
    preds = get_predictions(config, trainer, "eigen_benchmark")

    metrics = evaluator.run(preds, labels)

    split = "KEB"
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics.write(os.path.join(config.project_path, config.run_name, split + "_metrics_" + date_time + ".md"))
    print(metrics)


if __name__ == "__main__":
    main()
