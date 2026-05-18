import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric

try:
    from kaolin.metrics.pointcloud import chamfer_distance
except ImportError:
    print("Kaolin not found, edge metrics will not be available")


from .projection import BackprojectDepth
from kornia.filters import canny
from scipy import ndimage
from math import log


class Metrics(nn.Module):
    """Class for computing depth estimation metrics."""

    def __init__(self, min_depth: float = 0.1, max_depth: float = 100.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.metrics = nn.ModuleDict(
            {
                "MAE": MAE(),
                "RMSE": RMSE(),
                "LogSI": ScaleInvariant(mode="log"),
                "AbsRel": AbsRel(),
                "Acc": DeltaAcc(delta=1.25),
            }
        )

    def preprocess(self, pred, target):
        """Preprocess depth maps for metric computation.

        Computes the scale factor to align predicted and target depths,
        and creates a valid mask for evaluation.

        Args:
            pred (torch.Tensor): Predicted depth map.
            target (torch.Tensor): Ground truth depth map.

        Returns:
            tuple:
                - scale (torch.Tensor): Scale factor to apply to predictions.
                - mask (torch.Tensor): Boolean mask of valid depth values.
        """
        mask = target > 0.1

        scale = torch.median(target[mask]) / torch.median(pred[mask])

        return scale, mask

    def compute_metrics_benchmark(self, pred_depth, target_depth):
        """Compute Kitti Benchmark depth prediction metrics.

        Args:
            pred_depth (torch.Tensor): Predicted depth map.
            target_depth (torch.Tensor): Ground truth depth map.

        Returns:
            dict: Dictionary containing benchmark metrics.
        """
        # Base errors (in meters)
        err = torch.abs(pred_depth - target_depth)
        err_sq = err**2

        # Inverse errors (in 1/km)
        err_inv = 1000 * torch.abs(1 / pred_depth - 1 / target_depth)
        err_inv_sq = err_inv**2

        # Log errors (in 100*log(m))
        err_log = 100 * (torch.log(pred_depth) - torch.log(target_depth))
        err_log_sq = err_log**2

        # Relative errors (in %)
        err_rel = 100 * (err / target_depth)
        err_rel_sq = 100 * (err_sq / target_depth**2)

        ratio = 100 * torch.max((target_depth / pred_depth), (pred_depth / target_depth))

        return {
            "MAE": err.mean(),
            "RMSE": torch.sqrt(err_sq.mean()),
            "InvMAE": err_inv.mean(),
            "InvRMSE": torch.sqrt(err_inv_sq.mean()),
            "LogMAE": torch.abs(err_log).mean(),
            "LogRMSE": torch.sqrt(err_log_sq.mean()),
            "LogSI": torch.sqrt(err_log_sq.mean() - err_log.mean() ** 2),
            "AbsRel": err_rel.mean(),
            "SqRel": err_rel_sq.mean(),
            "Acc": (ratio < 125).float().mean(),
        }

    def single_frame_metrics(self, pred, target):
        """Compute error metrics between predicted and ground truth depths.

        Args:
            pred: Depth prediction distribution object.
            target (torch.Tensor): Ground truth depth map.

        Returns:
            dict: Dictionary containing various metrics.
        """
        pred_depth = 1 / F.interpolate(pred.disp_mode_alpha(), size=target.shape[2:], mode="bilinear")
        scale, mask = self.preprocess(pred_depth, target)
        pred_depth = (pred_depth[mask] * scale).clamp(min=0.1, max=100)
        target_depth = target[mask].clamp(min=0.1, max=100)

        # Compute benchmark metrics
        metrics = self.compute_metrics_benchmark(pred_depth, target_depth)

        # Add likelihood metric
        # metrics["nlog_likelihood"] = pred.depth_nlog_likelihood(target, scale)[mask].mean()

        return metrics

    @torch.no_grad()
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute depth metrics for a dataset batch.

        :param pred: (Tensor) (b, 1, h, w) Scaled network depth predictions.
        :param target: (Tensor) (b, 1, h, w) Ground-truth LiDAR depth.
        :return: metrics: (TensorDict) Average metrics across batch.
        """
        min, max = self.min_depth, self.max_depth
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False).clamp(min, max)

        mask = target > 0
        target = target.where(mask, target.new_tensor(torch.nan))
        pred = pred.where(mask, pred.new_tensor(torch.nan))

        pred, target = pred.flatten(1), target.flatten(1)
        r = target.nanmedian(dim=1, keepdim=True).values / pred.nanmedian(dim=1, keepdim=True).values
        pred *= r

        pred.clamp_(min, max), target.clamp_(min, max)
        metrics = {k: metric(pred, target) for k, metric in self.metrics.items()}
        return metrics

    def forward(self, pred, target):
        return self.compute_metrics(pred.depth_mode_alpha().detach(), target.detach())

    # def forward(self, pred, target):
    #     """Compute metrics for a batch of predictions.

    #     Args:
    #         pred: Batch of depth prediction distribution objects.
    #         target (torch.Tensor): Batch of ground truth depth maps.

    #     Returns:
    #         dict: Dictionary containing batch-averaged metrics.
    #     """
    #     metrics = {}
    #     for b in range(target.shape[0]):
    #         single_metrics = self.single_frame_metrics(pred[b], target[b : b + 1])

    #         for k, v in single_metrics.items():
    #             metrics[k] = metrics.get(k, 0) + v

    #     for k, v in metrics.items():
    #         metrics[k] = v / target.shape[0]

    #     return metrics


MODES = {"raw", "log", "inv"}


class BaseMetric(Metric):
    higher_is_better = False
    full_state_update = False

    """Base class for depth estimation metrics."""

    def __init__(self, mode: str = "raw", **kwargs):
        super().__init__(**kwargs)
        assert mode in MODES
        self.mode: str = mode
        self.sf: int = {"raw": 1, "log": 100, "inv": 1000}[self.mode]

        self.add_state("metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _preprocess(self, input, /):
        """Convert input into log-depth or disparity."""
        if self.mode == "raw":
            pass
        elif self.mode == "log":
            input = input.log()
        elif self.mode == "inv":
            input = 1 / input.clip(min=1e-3)
        return input

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute an error metric for a single pair.

        :param pred: (Tensor) (b, n) Predicted depth.
        :param target: (Tensor) (b, n) Target depth.
        :return: (Tensor) (b,) Computed metric.
        """
        raise NotImplementedError

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Compute an error metric for a whole batch of predictions and update the state.

        :param pred: (Tensor) (b, n) Predicted depths masked with NaNs.
        :param target: (Tensor) (b, n) Target depths masked with NaNs.
        :return:
        """
        self.metric += self.sf * self._compute(self._preprocess(pred), self._preprocess(target)).sum()
        self.total += pred.shape[0]

    def compute(self) -> Tensor:
        """Compute the average metric given the current state."""
        return self.metric / self.total


class MAE(BaseMetric):
    """Compute the mean absolute error."""

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).abs().nanmean(dim=1)


class RMSE(BaseMetric):
    """Compute the root mean squared error."""

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).pow(2).nanmean(dim=1).sqrt()


class ScaleInvariant(BaseMetric):
    """Compute the scale invariant error."""

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        err = pred - target
        return (err.pow(2).nanmean(dim=1) - err.nanmean(dim=1).pow(2)).sqrt()


class AbsRel(BaseMetric):
    """Compute the absolute relative error."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sf = 100  # As %

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target).abs() / target).nanmean(dim=1)


class SqRel(BaseMetric):
    """Compute the absolute relative squared error."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sf = 100  # As %

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        return ((pred - target).pow(2) / target.pow(2)).nanmean(dim=1)


class DeltaAcc(BaseMetric):
    higher_is_better = True

    """Compute the accuracy for a given error threshold."""

    def __init__(self, delta: float, **kwargs):
        super().__init__(**kwargs)
        assert self.mode == "raw", "Accuracy should only be computed using raw depths."
        self.delta: float = delta
        self.sf = 100  # As %

    def _compute(self, pred: Tensor, target: Tensor) -> Tensor:
        thresh = torch.max(target / pred, pred / target)
        return (thresh < self.delta).nansum(dim=1) / thresh.nansum(dim=1)


# EIGEN
# -----------------------------------------------------------------------------
def metrics_eigen(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Compute Kitti Eigen depth prediction metrics.
    From Eigen (https://arxiv.org/abs/1406.2283)

    NOTE: The `sq_rel` error is incorrect! The correct error is `((err_sq ** 2) / target**2).mean()`
    We use the incorrect metric for backward compatibility with the common Eigen benchmark.
    This metric has been incorrectly reported since the benchmark was introduced.

    :param pred: (Tensor) (n,) Masked predicted depth.
    :param target: (Tensor) (n,) Masked ground truth depth.
    :return: (dict) Computed depth metrics.
    """
    err = torch.abs(pred - target)
    err_rel = err / target

    err_sq = err**2
    err_sq_rel = err_sq / target

    err_log_sq = (torch.log(pred) - torch.log(target)) ** 2

    thresh = torch.maximum((target / pred), (pred / target))

    return {
        "AbsRel": err_rel.mean(),
        "SqRel": err_sq_rel.mean(),
        "RMSE": torch.sqrt(err_sq.mean()),
        "LogRMSE": torch.sqrt(err_log_sq.mean()),
        "$\\delta < 1.25$": (thresh < 1.25).mean(),
        "$\\delta < 1.25^2$": (thresh < 1.25**2).mean(),
        "$\\delta < 1.25^3$": (thresh < 1.25**3).mean(),
    }


# -----------------------------------------------------------------------------


# BENCHMARK
# -----------------------------------------------------------------------------
def metrics_benchmark(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Compute Kitti Benchmark depth prediction metrics.
    From Kitti (https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_depth.zip devkit/cpp/evaluate_depth.cpp L19-120)

    Base errors are reported as `m`.
    Inv errors are reported as `1/km`.
    Log errors are reported as `100*log(m)`.
    Relative errors are reported as `%`.
    This roughly aligns the significant figures for all metrics.

    :param pred: (Tensor) (n,) Masked predicted depth.
    :param target: (Tensor) (n,) Masked ground truth depth.
    :return: (dict) Computed depth metrics.
    """
    err = torch.abs(pred - target)  # Units: m
    err_sq = err**2

    err_inv = 1000 * torch.abs(1 / pred - 1 / target)  # Units: 1/km
    err_inv_sq = err_inv**2

    # NOTE: This is a DIRECTIONAL error! This is required for the SI Log loss
    # Objective is to not penalize the prediction if the errors are consistently in the same direction.
    # I.e. if the prediction could be aligned by applying a constant scale factor.
    err_log = 100 * (torch.log(pred) - torch.log(target))  # Units: log(m)*100
    err_log_sq = err_log**2

    err_rel = 100 * (err / target)  # Units: %
    err_rel_sq = 100 * (err_sq / target**2)

    return {
        "MAE": err.mean(),
        "RMSE": torch.sqrt(err_sq.mean()),
        "InvMAE": err_inv.mean(),
        "InvRMSE": torch.sqrt(err_inv_sq.mean()),
        "LogMAE": torch.abs(err_log).mean(),
        "LogRMSE": torch.sqrt(err_log_sq.mean()),
        "LogSI": torch.sqrt(err_log_sq.mean() - err_log.mean() ** 2),
        "AbsRel": err_rel.mean(),
        "SqRel": err_rel_sq.mean(),
    }


# -----------------------------------------------------------------------------


# POINTCLOUD
# -----------------------------------------------------------------------------
def _metrics_pointcloud(pred: Tensor, target: Tensor, th: float) -> tuple[Tensor, Tensor]:
    """Helper to compute F-Score and IoU with different correctness thresholds."""
    P = (pred < th).float().mean()  # Precision - How many predicted points are close enough to GT?
    R = (target < th).float().mean()  # Recall - How many GT points have a predicted point close enough?
    if (P < 1e-3) and (R < 1e-3):
        return P, P  # No points are correct.

    f = 2 * P * R / (P + R)
    iou = P * R / (P + R - (P * R))
    return f, iou


def metrics_pointcloud(pred: Tensor, target: Tensor, mask: Tensor, K: Tensor) -> dict[str, float]:
    """Compute pointcloud-based prediction metrics.
    From Ornek: (https://arxiv.org/abs/2203.08122)

    These metrics are computed on the GPU, since Chamfer distance has quadratic complexity.
    Following the original _paper, we set the default threshold of a correct point to 10cm.
    An extra threshold is added at 20cm for informative purposes, but is not typically reported.

    :param pred: (Tensor) (h, w) Predicted depth.
    :param target: (Tensor) (h, w) Ground truth depth.
    :param mask: (Tensor) (h, w) Mask of valid pixels.
    :param K: (Tensor) (4, 4) Camera intrinsic parameters.
    :return: (dict) Computed depth metrics.
    """
    K_inv = K.inverse()[None]
    backproj = BackprojectDepth(pred.shape).to(pred.device)
    pred_pts = backproj(pred[None, None], K_inv)[:, :3, mask.flatten()]
    target_pts = backproj(target[None, None], K_inv)[:, :3, mask.flatten()]

    pred_nn, target_nn = chamfer_distance(pred_pts.permute(0, 2, 1), target_pts.permute(0, 2, 1))
    pred_nn, target_nn = pred_nn.sqrt(), target_nn.sqrt()

    f1, iou1 = _metrics_pointcloud(pred_nn, target_nn, th=0.1)
    f2, iou2 = _metrics_pointcloud(pred_nn, target_nn, th=0.2)
    return {
        "Chamfer": pred_nn.mean() + target_nn.mean(),
        "F-Score": 100 * f1,
        "IoU": 100 * iou1,
        "F-Score-20": 100 * f2,
        "IoU-20": 100 * iou2,
    }


# -----------------------------------------------------------------------------


# EDGES
# -----------------------------------------------------------------------------
def metrics_ibims(pred: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    """Compute edge-based prediction metrics.
    From IBIMS: (https://arxiv.org/abs/1805.01328v1)

    The main metrics of interest are the edge accuracy and completeness. However, we also provide the directed error.
    Edge accuracy measures how close the predicted edges are wrt the ground truth edges.
    Meanwhile, edge completeness measures how close the ground-truth edges are from the predicted ones.

    :param pred: (Tensor) (h, w) Predicted depth.
    :param target: (Tensor) (h, w) Ground truth depth.
    :param mask: (Tensor) (h, w) Mask of valid & edges pixels.
    :param K: (Tensor) (4, 4) Camera intrinsic parameters.
    :return: (dict) Computed depth metrics.
    """
    th_dir = 10  # Plane at 10 meters
    pred_dir = torch.where(pred <= th_dir, 1, 0)
    target_dir = torch.where(target <= th_dir, 1, 0)
    err_dir = pred_dir - target_dir

    th_edges = 10
    D_target = torch.tensor(
        ndimage.distance_transform_edt(torch.logical_not(mask))
    )  # Distance of each pixel to ground truth edges
    pred_edges = extract_edges(pred)
    D_pred = torch.tensor(
        ndimage.distance_transform_edt(torch.logical_not(pred_edges))
    )  # Distance of each pixel to predicted edges
    pred_edges = pred_edges.bool() & (D_target < th_edges)  # Predicted edges close enough to real ones.

    return {
        "DirAcc": 100 * (err_dir == 0).float().mean(),  # Accurate order
        "Dir (-)": 100 * (err_dir == 1).float().mean(),  # Pred depth was underestimated
        "Dir (+)": 100 * (err_dir == -1).float().mean(),  # Pred depth was overestimated
        "EdgeAcc": D_target[pred_edges].mean() if pred_edges.sum() else th_edges,  # Distance from pred to target
        "EdgeComp": D_pred[mask].mean() if pred_edges.sum() else th_edges,  # Distance from target to pred
    }


# -----------------------------------------------------------------------------
def extract_edges(
    depth: Tensor,
) -> Tensor:
    """Detect edges in a dense LiDAR depth map.

    :param depth: (Tensor) (h, w, 1) Dense depth map to extract edges.
    :param preprocess: (str) Additional depth map post-processing. (log, inv, none)
    :param sigma: (int) Gaussian blurring sigma.
    :param mask: (Optional[Tensor]) Optional boolean mask of valid pixels to keep.
    :param use_canny: (bool) If `True`, use `Canny` edge detection, otherwise `Sobel`.
    :return: (Tensor) (h, w) Detected depth edges in the image.
    """

    depth = depth.squeeze()
    depth = depth.log()
    edges = canny(depth[None, None])[1]
    return edges.squeeze()


def metrics_likelihood(pred: Tensor, target_mask: Tensor, mask: Tensor, scale) -> dict[str, float]:
    """Compute likelihood-based prediction metrics.

    :param pred: (Tensor) (h, w) Predicted depth.
    :param target: (Tensor) (h, w) Ground truth depth.
    :param mask: (Tensor) (h, w) Mask of valid pixels.
    :return: (dict) Computed depth metrics.
    """
    target_scaled_inv = scale / target_mask
    mu = pred.mu[0, :, mask]
    sigma = pred.sigma[0, :, mask]
    w = pred.alpha[0, :, mask]  # 2 N

    mix_nll = mixture_likelihood(mu, sigma, w, target_scaled_inv)
    component_nll = component_likelihood(mu, sigma, w, target_scaled_inv)
    var, optimal_nll = optimal_likelihood(mu, w, target_scaled_inv)
    return {
        "NLL_Mix": mix_nll,
        "NLL_Comp": component_nll,
        "NLL_Opt": optimal_nll,
        "Var_Opt": var,
    }


def mixture_likelihood(mu, sigma, w, label):
    """Compute negative log-likelihood of a gaussian mixture distribution using logsumexp.

    :param mu: (Tensor) (C, N) Means of the mixture components.
    :param sigma: (Tensor) (C, N) Standard deviations of the mixture components.
    :param w: (Tensor) (C, N) Weights of the mixture components.
    :param label: (Tensor) (N,) Ground truth values.
    :return: (Tensor) Negative log-likelihood.
    """
    C, N = mu.shape
    label = label.unsqueeze(0).expand(C, -1)  # (C, N)

    log_probs = -0.5 * (((label - mu) / sigma) ** 2 + 2 * torch.log(sigma) + log(2 * torch.pi))  # (C, N)
    log_weights = torch.log(w + 1e-8)  # (C, N)
    nll = -torch.logsumexp(log_weights + log_probs, dim=0).mean()  # (N,)

    return nll


def component_likelihood(mu, sigma, w, label):
    """Compute negative log-likelihood of the most likely gaussian component.

    :param mu: (Tensor) (C, N) Means of the mixture components.
    :param sigma: (Tensor) (C, N) Standard deviations of the mixture components.
    :param label: (Tensor) (N,) Ground truth values.
    :return: (Tensor) Negative log-likelihood.
    """
    C, N = mu.shape
    label = label.unsqueeze(0).expand(C, -1)  # (C, N)
    w_bin = F.one_hot(torch.argmax(w, dim=0), num_classes=w.shape[0]).T.bool()
    log_probs = -0.5 * (((label - mu) / sigma) ** 2 + 2 * torch.log(sigma) + log(2 * torch.pi))  # (C, N)
    nll = -torch.sum(w_bin * log_probs, dim=0).mean()
    return nll


def optimal_likelihood(mu, w, label):
    w_bin = F.one_hot(torch.argmax(w, dim=0), num_classes=w.shape[0]).T.bool()
    pred_mu = torch.sum(w_bin * mu, dim=0)

    # Compute variance (to be averaged across dataset, then sqrt for std)
    residuals = label - pred_mu
    variance = (residuals**2).mean()

    # Compute average negative log-likelihood using the variance
    sigma = torch.sqrt(variance + 1e-8)
    log_likelihood = -0.5 * (((label - pred_mu) / sigma) ** 2 + 2 * torch.log(sigma) + log(2 * torch.pi))
    nll = -log_likelihood.mean()

    return variance, nll
