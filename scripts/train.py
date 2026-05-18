import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
import torch
import numpy as np
from sob import Trainer
from sob import TrainingConfig
from sob.evaluator import Evaluator, get_predictions


def main():
    """Main entry point for training depth and pose estimation models.

    This function:
    1. Sets PyTorch precision for better performance
    2. Loads configuration from command line arguments or config file
    3. Initializes the Trainer with the configuration
    4. Runs either the overfit routine (for debugging) or the full training

    The configuration parameters determine all aspects of the training,
    including model architecture, datasets, optimization, and logging.
    """
    torch.set_float32_matmul_precision("high")  # medium (bfloat16), high (tf32), highest (fp32)
    config = TrainingConfig.from_args()
    config.save()
    trainer = Trainer(config)
    if config.overfit:
        trainer.overfit()
    else:
        trainer.run()

    # Run validatiion on eigen_zhou valiation set

    preds = get_predictions(config, trainer, "test")
    labels = np.load(f"{config.data_path}/Kitti/splits/eigen_zhou/targets_test.npz", allow_pickle=True)

    metrics = Evaluator().run(preds, labels)
    metrics.write(os.path.join(config.project_path, config.run_name, "KEZ_metrics_last.md"))
    print(metrics)


if __name__ == "__main__":
    main()
