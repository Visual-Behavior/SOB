from typing import Optional, Sequence
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize
from .config import TrainingConfig
from .datasets import KittiDataset
from .trainer import Trainer
from .utils import track, Table
from .metrics import metrics_benchmark, metrics_eigen, metrics_pointcloud, metrics_ibims, metrics_likelihood
from .distribution import MixtureDistribution


def get_predictions(config: TrainingConfig, trainer: Trainer, split: str):
    # Use the appropriate split; ground truth is not used here
    dataset = KittiDataset(split=split, sources=tuple())
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False
    )

    dist_predictions = []

    for batch in track(dataloader):
        pred_dist = trainer.inference_step(batch["target"].cuda())
        # Get disparity predictions from the model.
        dist_predictions.append(pred_dist.cpu())

    return dist_predictions


class Evaluator:
    """Class to evaluate unscaled network depth predictions.

    NOTE:
        - Pointcloud metrics can only be computed when camera intrinsics are present.
        - IBIMS metrics can only be computed when depth edges are present.

    :param mode: (str) Evaluation mode. {stereo, mono}
    :param metrics: (list[str]) List of metric sets to compute. {eigen, benchmark, pointcloud, ibims}
    :param min: (float) Min ground-truth depth to evaluate.
    :param max: (float) Max ground-truth depth to evaluate.
    :param use_eigen_crop: (bool) If `True` use border cropping. Should only be used with the Kitti Eigen split.
    """

    STEREO_SF = 5.4  # Fixed Kitti scaling, given that we train using an arbitrary baseline of 0.1 vs. the real 54cm.

    def __init__(
        self,
        mode: str = "mono",
        metrics: Sequence[str] = ("benchmark"),
        min: float = 1e-3,
        max: float = 100,
        use_eigen_crop: bool = False,
    ):
        self.mode = mode
        self.metrics = metrics
        self.min = min
        self.max = max
        self.use_eigen_crop = use_eigen_crop

    @staticmethod
    def _get_eigen_mask(shape: tuple[int, int]) -> Tensor:
        """Helper to get the border masking introduced by Eigen."""
        h, w = shape
        crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w], dtype=int)
        mask = np.zeros((h, w), dtype=bool)
        mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
        return mask

    def _get_mask(self, target: Tensor) -> Tensor:
        """Helper to mask ground-truth depth based on the selected range and Eigen crop."""
        mask = target > self.min
        if self.max:
            mask &= target < self.max
        if self.use_eigen_crop:
            mask &= self._get_eigen_mask(target.shape)
        return mask

    def _get_ratio(self, pred: Tensor, target: Tensor) -> float:
        """Helper to get the prediction scaling ratio based on the evaluation mode. Stereo=fixed, Mono=median."""
        return self.STEREO_SF if self.mode == "stereo" else float(torch.median(target) / torch.median(pred))

    def _upsample(self, pred: Tensor, target: Tensor) -> Tensor:
        """Helper to upsample the prediction to the full target resolution."""
        h, w = target.shape
        pred = Resize((h, w))(pred[None]).squeeze(0)
        return pred

    def _eval_single(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        K: Optional[Tensor],
        cat: Optional[str],
        subcat: Optional[str],
        metrics: Sequence[str],
    ):
        """Helper to compute metrics from a single prediction."""
        if mask.sum() == 0:
            return {}

        if isinstance(pred, MixtureDistribution):
            pred_dist = pred.clone()
            pred = pred.depth_mode_alpha()[0, 0]

        pred_mask, target_mask = pred[mask], target[mask]

        r = self._get_ratio(pred_mask, target_mask)
        pred, pred_mask = (r * pred).clip(self.min, self.max), (r * pred_mask).clip(self.min, self.max)

        ms = {}
        if cat:
            ms["Cat"] = str(cat)
        if subcat:
            ms["SubCat"] = str(subcat)

        for m in metrics:
            if m == "eigen":
                ms.update(metrics_eigen(pred_mask, target_mask))
            elif m == "benchmark":
                ms.update(metrics_benchmark(pred_mask, target_mask))
            elif m == "pointcloud":
                ms.update(metrics_pointcloud(pred, target, mask, K))
            elif m == "ibims":
                ms.update(metrics_ibims(pred, target, mask))
            elif m == "likelihood":
                ms.update(metrics_likelihood(pred_dist, target_mask, mask, r))

        return ms

    def run(self, preds, data):
        """Compute evaluation metrics over a whole dataset, specified by the target `data`.

        :param preds: (ndarray) (b, h, w) Unscaled disparity predictions, where `b=len(dataset)`.
        :param data: (ArrDict) Network targets (depth, *K, *edge, *cat, *subcat) loaded from an `.npz` file.
        :return: (list(Metrics)) Computed metrics for each dataset item.
        """
        device = preds[0].device
        targets, Ks, edges = data["depth"], torch.tensor(data.get("K"), device=device), data.get("edge")
        cats, subcats = data.get("cat"), data.get("subcat")

        if (a := len(preds)) != (b := len(targets)):
            raise ValueError(f"Non-matching preds and targets! ({a} vs. {b})")

        ts = [torch.tensor(t, dtype=torch.float32, device=device) for t in targets]
        ms = [self._get_mask(t) for t in ts]

        print("Upsampling predictions...")
        if isinstance(preds[0], MixtureDistribution):
            ps = [p.upsample(*t.shape) for p, t in zip(preds, ts)]
        else:
            ps = [1 / (self._upsample(p, t)) for p, t in zip(preds, ts)]  # Convert disparity to depth and upsample

        if Ks is None:
            Ks = [None] * len(ts)
        if cats is None:
            cats = [None] * len(ts)
        if subcats is None:
            subcats = [None] * len(ts)

        print("\n-> Computing metrics...")
        metrics = [
            self._eval_single(p, t, m, K, c1, c2, [m for m in self.metrics if m != "ibims"])
            for p, t, m, K, c1, c2 in zip(track(ps), ts, ms, Ks, cats, subcats)
        ]
        if edges is not None:
            print("\n-> Computing edges-based metrics...")
            ms = [m1 & m2 for m1, m2 in zip(ms, edges)]
            metrics_edge = [
                self._eval_single(p, t, m, K, c1, c2, self.metrics)
                for p, t, m, K, c1, c2 in zip(track(ps), ts, ms, Ks, cats, subcats)
            ]
            metrics_edge = [{f"{k}-Edges": v for k, v in m.items()} for m in metrics_edge]
            metrics = [{**m1, **m2} for m1, m2 in zip(metrics, metrics_edge)]
        mean_metrics = {k: float(torch.stack([d[k] for d in metrics if k in d]).mean()) for k in metrics[0]}
        if "Var_Opt" in mean_metrics:
            # replace by stddev
            var = mean_metrics.pop("Var_Opt")
            mean_metrics["Sigma_Opt"] = var**0.5

        return Table(mean_metrics)
