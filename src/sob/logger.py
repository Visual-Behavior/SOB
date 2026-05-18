from torch.utils.tensorboard import SummaryWriter
from matplotlib.cm import get_cmap
import torch
import os
from collections import defaultdict


class Logger:
    """Logger class for training depth and pose estimation models.

    This class handles logging metrics, images, and distributions to TensorBoard
    during training and validation.

    Attributes:
        writers (dict): Dictionary of TensorBoard SummaryWriters for each mode.
        num_images (int): Number of images to log per batch.
        avg_window (int): Number of steps over which to average losses and metrics during training.
        accumulators (dict): Dictionary of accumulators for different value types.
        step_counters (dict): Dictionary of step counters for different modes.
    """

    def __init__(self, config, avg_window=50):
        """Initialize the logger.

        Args:
            run_path (str): Directory path where logs will be saved.
            batch_size (int): Batch size used in training, used to determine
                the number of images to log.
            avg_window (int, optional): Number of steps to average over during training. Defaults to 50.
        """
        run_path = config.run_path
        batch_size = config.batch_size
        self.is_mixture = config.components > 1

        self.writers = {
            "train": SummaryWriter(os.path.join(run_path, "train")),
            "val": SummaryWriter(os.path.join(run_path, "val")),
        }
        self.num_images = min(batch_size, 4)
        self.avg_window = avg_window
        self.accumulators = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.step_counters = defaultdict(int)

    def accumulate(self, data_dict, mode="train"):
        """Accumulate data for later logging.

        Args:
            data_dict (dict): Nested dictionary with categories as keys
                (e.g., {'losses': {...}, 'metrics': {...}, 'grads': {...}})
            mode (str): Logging mode ('train' or 'val')
        """
        if not data_dict:
            return

        # Accumulate all values
        for data_type, values in data_dict.items():
            if values is not None:
                for name, value in values.items():
                    self.accumulators[mode][data_type][name] += value

        self.step_counters[mode] += 1

    def flush(self, step, mode="train"):
        """Calculate averages and log accumulated data.

        Args:
            step (int): Global step for logging
            mode (str): Logging mode ('train' or 'val')

        Returns:
            dict: The averaged data that was logged (or None if nothing was logged)
        """
        # Return immediately if no data accumulated
        if self.step_counters[mode] == 0:
            return None

        # Calculate averages for all accumulated data
        averaged_data = defaultdict(dict)
        for data_type, values in self.accumulators[mode].items():
            for name, value in values.items():
                averaged_data[data_type][name] = value / self.step_counters[mode]

        # Log to TensorBoard
        for data_type, values in averaged_data.items():
            for name, value in values.items():
                self.writers[mode].add_scalar(f"{mode}_{data_type}/{name}", value, global_step=step)

        # Reset accumulators and counter for this mode
        self.accumulators[mode] = defaultdict(lambda: defaultdict(float))
        self.step_counters[mode] = 0

        return dict(averaged_data)  # Convert from defaultdict to regular dict

    def log_images(self, images, step, mode="train"):
        """Log images to TensorBoard.

        Different types of images (target, mu_c, error, etc.) are handled differently.
        Distribution objects are handled by log_distribution.

        Args:
            images (dict): Dictionary of image names and tensors.
            step (int): Global step for logging.
            mode (str, optional): Logging mode, either "train" or "val". Defaults to "train".
        """
        for name, image in images.items():
            if name == "distribution":
                self.log_distribution(image, step, mode)
                continue
            for i in range(self.num_images):
                if name == "target" or name == "mu_c":
                    self.writers[mode].add_image(f"{name}/{i}", image[i].cpu(), step, dataformats="CHW")
                elif name == "error" or name == "l1":
                    self.writers[mode].add_image(f"{name}/{i}", normalize_image(image[i].cpu()), step)
                elif name == "w":
                    self.writers[mode].add_image(f"{name}/{i}", attention_image(image[i].cpu()), step)
                elif name == "mask":
                    self.writers[mode].add_image(f"{name}/{i}", 1 - image[i].float().cpu(), step, dataformats="CHW")
                elif name == "mask_depth":
                    self.writers[mode].add_image(f"{name}/{i}", depth_to_image(image[i].cpu()), step)
                elif name == "masked_loss":
                    self.writers[mode].add_image(f"{name}/{i}", normalize_image(image[i].cpu(), q=0.99), step)

    def log_distribution(self, distribution, step, mode="train"):
        """Log depth distribution visualizations to TensorBoard.

        Visualizes various aspects of a depth distribution, including mean depth,
        mode depth, sigma, alpha, and component differences.

        Args:
            distribution: Depth distribution object with prediction properties.
            step (int): Global step for logging.
            mode (str, optional): Logging mode, either "train" or "val". Defaults to "train".
        """
        depth_mean = distribution.depth_mean()
        depth_mode = distribution.depth_mode_alpha()
        if distribution.mu.shape[1] > 1:
            diff_mu = distribution.diff_mu_rel()
        else:
            diff_mu = torch.zeros_like(distribution.mu)
        alpha = distribution.alpha
        sigma = (alpha * distribution.sigma).sum(dim=1)

        self.writers[mode].add_scalar("stats/depth_spatial_mean", distribution.depth_spatial_mean(), global_step=step)
        self.writers[mode].add_scalar("stats/sigma_spatial_mean", distribution.sigma_spatial_mean(), global_step=step)


        for i in range(self.num_images):
            self.writers[mode].add_image(f"depth_mean/{i}", depth_to_image(depth_mean[i]), global_step=step)
            if self.is_mixture:
                self.writers[mode].add_image(f"depth_mode/{i}", depth_to_image(depth_mode[i]), global_step=step)
                self.writers[mode].add_image(f"sigma/{i}", normalize_image(sigma[i]), global_step=step)
                self.writers[mode].add_image(f"alpha/{i}", attention_image(alpha[i, 1]), global_step=step)
                self.writers[mode].add_image(f"diff_mu/{i}", diff_mu[i] / diff_mu[1].amax(), global_step=step)

    def log_config(self, config):
        """Log hyperparameters config to TensorBoard.

        Args:
            config: Configuration object with hyperparameters.
        """
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            for type in [int, float, str, bool]:
                if isinstance(value, type):
                    break
            else:
                # If value is not any of the basic types, convert to string
                config_dict[key] = str(value)

        self.writers["train"].add_hparams(config_dict, {})


def depth_to_image(depth):
    """Convert depth tensor to RGB image for visualization.

    Uses the turbo colormap with inverse depth mapping for better visualization.

    Args:
        depth (torch.Tensor): Depth map tensor.

    Returns:
        numpy.ndarray: RGB image in CHW format for TensorBoard.
    """
    # use inverse depth with turbo colormap
    non_zero = depth > 1e-6
    min_depth = depth[non_zero].min() if non_zero.sum() > 0 else 1
    max_depth = depth.max() if non_zero.sum() > 0 else 1

    idepth = 1 / depth.clip(min_depth, max_depth)
    idepth = (idepth - 1 / max_depth) / (1 / min_depth - 1 / max_depth)
    img = get_cmap("turbo")(idepth.squeeze().detach().cpu().numpy())[..., :3].transpose(2, 0, 1)
    img[:, depth.squeeze().detach().cpu().numpy() < 1e-6] = 1
    return img


def normalize_image(x, cmap="turbo", q=0.95):
    """Normalize tensor to [0,1] range and apply colormap.

    Args:
        x (torch.Tensor): Input tensor to normalize.
        cmap (str, optional): Matplotlib colormap name. Defaults to "turbo".
        q (float, optional): Quantile for max value. Defaults to 0.95.

    Returns:
        numpy.ndarray: RGB image in CHW format for TensorBoard.
    """
    ma = float(torch.quantile(x, q).cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    image = torch.clip((x - mi) / d, 0, 1)
    return get_cmap(cmap)(image.squeeze().detach().cpu().numpy())[..., :3].transpose(2, 0, 1)


def attention_image(x, cmap="RdBu_r"):
    """Convert attention/alpha values to RGB image.

    Args:
        x (torch.Tensor): Attention/alpha values.
        cmap (str, optional): Matplotlib colormap name. Defaults to "RdBu_r".

    Returns:
        numpy.ndarray: RGB image in CHW format for TensorBoard.
    """
    return get_cmap(cmap)(x.squeeze().detach().cpu().numpy())[..., :3].transpose(2, 0, 1)
