import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from collections import defaultdict

from .logger import Logger
from .metrics import Metrics
from .losses import GeometricLoss
from .datasets import KittiDataset
from .config import TrainingConfig
from .networks.model import DepthModel, PoseModel
from .distribution import GaussianMixture
from .utils import track


class Trainer:
    """Trainer class for depth and pose estimation models.

    This class handles training, validation, and inference for depth and pose
    estimation models, including setup of dataloaders, models, loss functions,
    and optimization routines.

    Attributes:
        config (TrainingConfig): Configuration object with training parameters.
        run_path (str): Path where model checkpoints and logs will be saved.
        accelerator (Accelerator): Accelerator for distributed training.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        depth_model (DepthModel): Model for depth estimation.
        pose_model (PoseModel): Model for pose estimation.
        loss (GeometricLoss): Loss function for training.
        metrics (Metrics): Metrics for evaluation.
        distribution (class): Distribution class for depth uncertainty modeling.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the Trainer with the given configuration.

        Args:
            config (TrainingConfig): Configuration object with training parameters.
        """
        self.config = config
        self.run_path = config.run_path
        self.accelerator = Accelerator()
        prep = self.accelerator.prepare
        self.train_loader, self.val_loader = prep(*self.get_dataloaders())

        self.depth_model = prep(
            nn.SyncBatchNorm.convert_sync_batchnorm(
                DepthModel(
                    config.encoder,
                    num_layers=config.num_layers,
                    components=config.components,
                    decoder=config.decoder,
                    sigma_type=config.sigma_type,
                    pretrain_path=config.pretrain_path,
                )
            )
        )
        self.pose_model = prep(
            nn.SyncBatchNorm.convert_sync_batchnorm(
                PoseModel(num_layers=config.num_layers, pretrain_path=config.pretrain_path)
            )
        )
        self.loss = GeometricLoss(config)
        if "color" in config.sigma_type:
            self.loss = prep(self.loss)

        if not config.no_compile:
            self.depth_model = torch.compile(self.depth_model)
            self.pose_model = torch.compile(self.pose_model)
            self.loss = torch.compile(self.loss)

        self.metrics = prep(Metrics())

        self.distribution = GaussianMixture

        params = list(self.depth_model.parameters()) + list(self.pose_model.parameters())

        self.optimizer = prep(torch.optim.Adam(params, lr=self.config.learning_rate))
        self.scheduler = prep(
            torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
            )
        )

        self.logger = Logger(config)

        self.epoch = 0
        self.total_steps = self.config.epochs * len(self.train_loader)
        self.best_metric = None

        if config.load_run is not None:
            self.load_state()

        self.global_step = self.epoch * len(self.train_loader)

    def run(self):
        """Run the full training and validation loop for the configured number of epochs.

        First validates on epoch 0, then runs training and validation for each epoch,
        saving the model state after each validation.
        """
        self.run_val()
        for epoch in range(self.epoch, self.config.epochs):
            self.run_train()
            metric = self.run_val()
            self.scheduler.step()
            self.save_state(metric, epoch)

    def overfit(self):
        """Run model training on a single batch repeatedly for overfitting test.

        This method is used for debugging by training on the same batch for 10001 steps,
        logging losses, gradients and images at regular intervals.
        """
        inputs, targets = next(iter(self.train_loader))
        for step in track(range(10001)):
            logs = self.train_step(inputs, targets)
            self.logger.log_dict(logs, step=step, mode="train")
            if step % 500 == 0 or step in [100, 250]:
                self.logger.log_images(logs["images"], step=step, mode="train")

    def run_train(self):
        """Run a full training epoch.

        Sets models to training mode, performs training steps on all batches in the
        training dataloader, and logs losses, gradients and images at specified intervals.
        """
        self.depth_model.train()
        self.pose_model.train()

        for i, inputs in enumerate(track(self.train_loader)):
            inputs["is_first"] = self.global_step == 0
            inputs["is_train"] = True
            logs, images = self.train_step(inputs, None)

            # Accumulate logs for averaging
            self.logger.accumulate(logs, mode="train")

            # Flush accumulated logs every avg_window steps
            if (i + 1) % self.logger.avg_window == 0:
                self.logger.flush(step=self.global_step, mode="train")

            # Log images at specific steps
            if self.global_step in [0, 100, 300, 600, 1000, 1500, 2000] or i == 0:
                self.logger.log_images(images, step=self.global_step, mode="train")

            self.global_step += 1

        # Force flush any remaining accumulated data at the end of epoch
        self.logger.flush(step=self.global_step, mode="train")

    def run_val(self):
        """Run validation on a subset of the validation data.

        Sets models to evaluation mode, performs validation steps on the validation
        dataloader (up to num_val_steps batches), and logs losses, metrics and images.

        Returns:
            float: Average validation metric used for model selection.
        """
        with torch.no_grad():
            self.depth_model.eval()
            self.pose_model.eval()
            val_iter = iter(self.val_loader)

            num_val_steps = 1600 // self.config.batch_size

            for step in track(range(num_val_steps)):
                inputs, targets = next(val_iter)
                inputs["is_first"] = False
                inputs["is_train"] = False
                logs, images = self.validation_step(inputs, targets)

                # Log images only on first step
                if step == 0:
                    self.logger.log_images(images, step=self.global_step, mode="val")
                # Accumulate validation metrics and losses
                self.logger.accumulate(logs, mode="val")

            # Force flush all accumulated validation data at the end
            averaged_data = self.logger.flush(step=self.global_step, mode="val")

        return averaged_data["metrics"][self.config.checkpoint_metric]

    def forward_step(self, target, sources, baseline=None):
        """Forward pass through both depth and pose networks.

        Args:
            target (torch.Tensor): Target image to process.
            sources (dict): Dictionary of source images keyed by source offset (-1, 1, 0 for stereo).
            baseline (torch.Tensor): Stereo baseline distances.

        Returns:
            tuple:
                - depth_output: Output from the depth model.
                - pose_output (dict): Dictionary of pose outputs keyed by source type.
        """
        depth_output = self.depth_model(target)
        for i in range(len(depth_output)):
            if "sigma" not in depth_output[i]:
                depth_output[i]["sigma"] = torch.ones_like(depth_output[i]["mu"])
            if "alpha" not in depth_output[i]:
                depth_output[i]["alpha"] = torch.ones_like(depth_output[i]["mu"])

        # Handle pose outputs based on available sources
        pose_output = {}
        for source in self.config.sources:
            if source == 0:
                # For stereo, create transform matrix using baseline
                B = target.shape[0]
                device = target.device
                # Create identity rotation and baseline translation
                pose_output[source] = (
                    torch.zeros(B, 3, device=device),  # No rotation
                    torch.stack(
                        [baseline, torch.zeros(B, device=device), torch.zeros(B, device=device)], dim=1
                    ),  # Translation in x
                )
            elif source < 0:  # Previous frame (negative offset)
                pose_output[source] = self.pose_model(torch.cat([sources[source], target], dim=1))
            else:  # Next frame (positive offset)
                pose_output[source] = self.pose_model(torch.cat([target, sources[source]], dim=1))

        return depth_output, pose_output

    def train_step(self, inputs, targets=None):
        """Perform a single training step.

        Computes forward pass, loss, and performs backpropagation and optimization.

        Args:
            inputs (dict): Dictionary containing input data including target and source images.

        Returns:
            dict: Dictionary containing:
                - losses: Loss values.
                - grads: Gradient norms.
                - images: Visual outputs for logging.
        """
        target = inputs["target_aug"]
        sources = {k: inputs[f"source_{k}_aug"] for k in self.config.sources}
        baseline = inputs.get("baseline", None)
        depth_output, pose_output = self.forward_step(target, sources, baseline)
        loss, losses, images = self.loss(inputs, depth_output, pose_output)

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()

        pred_dist = self.distribution(
            depth_output[0]["mu"], depth_output[0]["sigma"], depth_output[0]["alpha"], not self.config.no_inv
        )

        images["distribution"] = pred_dist
        images["target"] = inputs["target"]
        if targets is not None:
            metrics = self.metrics(pred_dist, targets)
        else:
            metrics = None
        # "grads": self._get_grad_norms(),
        return {"losses": losses, "metrics": metrics}, images

    def validation_step(self, inputs, targets):
        """Perform a single validation step.

        Computes forward pass and loss without backpropagation or optimization.

        Args:
            inputs (dict): Dictionary containing input data including target and source images.
            targets (torch.Tensor): Ground truth depth maps for evaluation.

        Returns:
            dict: Dictionary containing:
                - losses: Loss values.
                - metrics: Evaluation metrics.
                - images: Visual outputs for logging.
        """
        target = inputs["target"]
        sources = {k: inputs[f"source_{k}"] for k in self.config.sources}
        baseline = inputs.get("baseline", None)
        depth_output, pose_output = self.forward_step(target, sources, baseline)
        _, losses, images = self.loss(inputs, depth_output, pose_output)

        pred_dist = self.distribution(
            depth_output[0]["mu"], depth_output[0]["sigma"], depth_output[0]["alpha"], not self.config.no_inv
        )
        images["distribution"] = pred_dist
        images["target"] = target
        metrics = self.metrics(pred_dist, targets)
        return {"losses": losses, "metrics": metrics}, images

    def inference_step(self, image):
        """Perform a single inference step for depth prediction.

        Args:
            image (torch.Tensor): Input image for depth prediction.

        Returns:
            Distribution: Depth distribution prediction.
        """
        with torch.no_grad():
            self.depth_model.eval()
            depth_output = self.depth_model(image)

            if "sigma" not in depth_output[0]:
                depth_output[0]["sigma"] = torch.ones_like(depth_output[0]["mu"]) * 0.01
            if "alpha" not in depth_output[0]:
                depth_output[0]["alpha"] = torch.ones_like(depth_output[0]["mu"][:, :1])

            pred_dist = self.distribution(
                depth_output[0]["mu"], depth_output[0]["sigma"], depth_output[0]["alpha"], not self.config.no_inv
            )
        return pred_dist

    def _get_grad_norms(self):
        """Compute gradient norms for the last layer of the depth decoder.

        Returns:
            dict: Dictionary of gradient norms for each channel and scale
        """
        grad_dict = {}

        # Get number of components from model
        k = self.config.components

        # Generate channel names dynamically based on number of components
        channel_names = []
        channel_names += (
            [f"mu_{i}" for i in range(k)] + [f"sigma_{i}" for i in range(k)] + [f"alpha_{i}" for i in range(k - 1)]
        )

        # Access the model through .module when using DDP
        model = self.depth_model.module if hasattr(self.depth_model, "module") else self.depth_model
        w_grad = torch.linalg.vector_norm(model.decoder.out_conv[-1].conv.weight.grad, dim=[1, 2, 3])
        for i, name in enumerate(channel_names):
            grad_dict[f"grad/{name}"] = w_grad[i].item()

        return grad_dict

    def save_state(self, metric, epoch):
        """Save model checkpoints.

        Saves the current model state to the 'last' checkpoint, and additionally
        to the 'best' checkpoint if the current metric is better than the previous best.

        Args:
            metric (float): Current value of the metric used for model selection.
            epoch (int): Current epoch number.
        """
        # always save as last checkpoint
        self.accelerator.save_state(os.path.join(self.run_path, "last"))

        if self.best_metric is None:
            self.best_metric = metric
        elif metric > self.best_metric:
            self.best_metric = metric
            self.accelerator.save_state(os.path.join(self.run_path, "best"))

    def load_state(self):
        """Load model checkpoint from a previous run.

        Loads either the 'best' or 'last' checkpoint based on configuration.
        """
        name = "best" if self.config.load_best else "last"
        self.accelerator.load_state(os.path.join(self.config.project_path, self.config.load_run, f"{name}"))
        self.epoch = self.scheduler.scheduler.last_epoch * (not self.config.finetune)
        self.scheduler.scheduler.last_epoch = self.epoch

    def get_dataloaders(self):
        """Create and configure train and validation data loaders.

        Returns:
            tuple:
                - train_dataloader (DataLoader): DataLoader for training data.
                - val_dataloader (DataLoader): DataLoader for validation data.
        """
        kwargs = {
            "data_path": self.config.data_path,
            "height": self.config.height,
            "width": self.config.width,
            "sources": self.config.sources,
        }
        if self.config.dataset == "kitti":
            dataset = KittiDataset

        self.train_dataset = dataset(**kwargs, split="train")
        self.val_dataset = dataset(**kwargs, split="val")
        # self.train_dataset = KittiRawDataset(split="eigen_zhou", mode="train", use_aug=True)
        # self.val_dataset = KittiRawDataset(split="eigen_zhou", mode="val")

        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            # sampler=PickedSampler(self.train_dataset, start_samples=[39183, 32828, 5962, 7284]),
        )

        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=SeededRandomSampler(self.val_dataset),
        )
        return train_dataloader, val_dataloader


class SeededRandomSampler:
    """Sampler that shuffles indices in a deterministic way with a fixed seed.

    This sampler is used to shuffle the validation set deterministically such that the same
    validation set is used for every training run.

    Attributes:
        num_samples (int): Number of samples in the dataset.
        shuffled_list (list): Pre-computed shuffled indices.
    """

    def __init__(self, data_source) -> None:
        """Initialize the sampler with a data source.

        Args:
            data_source: Dataset to sample from.
        """
        self.num_samples = len(data_source)
        generator = torch.Generator()
        generator.manual_seed(63984756)

        self.shuffled_list = torch.randperm(self.num_samples, generator=generator).tolist()

    def __iter__(self):
        yield from self.shuffled_list

    def __len__(self) -> int:
        return self.num_samples


class PickedSampler(SeededRandomSampler):
    """Sampler that starts with specific samples before continuing with shuffled indices.

    This sampler is an extension of SeededRandomSampler that always returns specific
    "start_samples" before the randomly shuffled indices.

    Attributes:
        shuffled_list (list): Pre-computed list of indices starting with start_samples.
    """

    def __init__(self, data_source, start_samples=[1440, 530, 4298, 1750]):
        """Initialize the sampler with a data source and starting samples.

        Args:
            data_source: Dataset to sample from.
            start_samples (list, optional): Sample indices to always include at the beginning.
                Defaults to [1440, 530, 4298, 1750].
        """
        super().__init__(data_source)
        self.shuffled_list = start_samples + self.shuffled_list

    def __len__(self) -> int:
        return self.num_samples + 4
