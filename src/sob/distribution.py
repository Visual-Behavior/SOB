from abc import abstractmethod
import torch
import torch.nn.functional as F
from math import sqrt


class MixtureDistribution:
    """Base class for mixture probability distributions.

    This class serves as a base for mixture distribution implementations,
    providing common methods for transforming between depth and inverse depth spaces,
    computing statistics, and evaluating likelihoods.

    Attributes:
        mu (torch.Tensor): Mean parameter for each component, shape [B, C, H, W].
        sigma (torch.Tensor): Scale parameter for each component, shape [B, C, H, W].
        alpha (torch.Tensor): Mixture weights, shape [B, C-1, H, W] or [B, C, H, W].
        inv (bool): If True, represents inverse depth (disparity) distribution.
    """

    def __init__(self, mu, sigma, alpha, inv=True):
        """Create a mixture distribution from the model outputs.

        Automatically detaches the tensors from the graph.

        Args:
            mu (torch.Tensor): Mean parameter for each component, shape [B, C, H, W].
            sigma (torch.Tensor): Scale parameter for each component, shape [B, C, H, W].
            alpha (torch.Tensor): Mixture weights, shape [B, C, H, W].
            inv (bool, optional): If True, represents inverse depth distribution.
                Defaults to True.
        """
        self.mu = mu.detach()
        self.sigma = sigma.detach()
        self.alpha = alpha.detach()
        self.inv = inv

    @property
    def device(self):
        return self.mu.device

    def to(self, *args, **kwargs):
        """Move distribution tensors to a device and/or dtype.

        Args:
            *args: Positional arguments forwarded to ``torch.Tensor.to``.
            **kwargs: Keyword arguments forwarded to ``torch.Tensor.to``.

        Returns:
            MixtureDistribution: A new distribution with moved tensors.
        """
        return self.__class__(
            self.mu.to(*args, **kwargs),
            self.sigma.to(*args, **kwargs),
            self.alpha.to(*args, **kwargs),
            self.inv,
        )

    def cpu(self):
        return self.to("cpu")

    def clone(self):
        return self.__class__(self.mu.clone(), self.sigma.clone(), self.alpha.clone(), self.inv)

    def transform(self, x):
        """Transform from depth to the space in which the distribution is defined.

        Args:
            x (torch.Tensor): Input depth values.

        Returns:
            torch.Tensor: Transformed values (inverse depth if inv=True, unchanged otherwise).
        """
        if self.inv:
            return 1 / (x + 1e-6)
        return x

    def transform_inv(self, x):
        """Transform from distribution space back to depth.

        Args:
            x (torch.Tensor): Input values in distribution space.

        Returns:
            torch.Tensor: Transformed values (unchanged if inv=True, inverse otherwise).
        """
        if self.inv:
            return x
        return 1 / (x + 1e-6)

    def nlog_jacobian(self, x, scale=1):
        """Compute the negative log Jacobian of the transform.

        Used for change of variable in probability distributions.

        Args:
            x (torch.Tensor): Input depth values.
            scale (float, optional): Scaling factor. Defaults to 1.

        Returns:
            torch.Tensor: Negative log Jacobian.
        """
        if self.inv:
            return torch.log(scale) - 2 * torch.log(x + 1e-6)
        return torch.log(scale)

    def depth_spatial_mean(self):
        """Compute the spatial mean of the expected depth.

        Returns:
            torch.Tensor: Mean depth value across spatial dimensions.
        """
        return self.depth_mean().mean()

    def sigma_spatial_mean(self):
        """Compute the spatial mean of sigma.

        Returns:
            torch.Tensor: Mean sigma value across spatial dimensions.
        """
        return self.sigma.mean()

    def depth_spatial_median(self):
        """Compute the spatial median of the expected depth.

        Returns:
            torch.Tensor: Median depth value across spatial dimensions.
        """
        return self.depth_mean().median()

    def depth_mean(self):
        """Compute the expected depth at each spatial location.

        Returns:
            torch.Tensor: Expected depth map, shape [B, 1, H, W].
        """
        return self.transform((self.alpha * self.mu).sum(1, True))

    def depth_mode_alpha(self):
        """Compute the mode of the depth distribution using rounded alpha values.

        Returns:
            torch.Tensor: Mode depth map, shape [B, 1, H, W].
        """
        return self.transform((self.alpha_bin * self.mu).sum(1, True))

    def upsample(self, H, W):
        """Upsample the distribution parameters to a specified resolution.

        Args:
            H (int): Target height.
            W (int): Target width.

        Returns:
            MixtureDistribution: Upsampled mixture distribution.
        """
        self.mu = F.interpolate(self.mu, size=(H, W), mode="bilinear", align_corners=False)
        self.sigma = F.interpolate(self.sigma, size=(H, W), mode="bilinear", align_corners=False)
        self.alpha = F.interpolate(self.alpha, size=(H, W), mode="bilinear", align_corners=False)
        return self

    @property
    def alpha_bin(self):
        return F.one_hot(torch.argmax(self.alpha, dim=1), num_classes=self.alpha.shape[1]).permute(0, 3, 1, 2).bool()

    def disp_mode_alpha(self):
        """Compute the mode of the disparity (inverse depth) using rounded alpha values.

        Returns:
            torch.Tensor: Mode disparity map, shape [B, 1, H, W].
        """
        return self.transform_inv((self.alpha_bin * self.mu).sum(1, True))

    def depth_mode_from_disp(self):
        """Compute the mode of the depth from disparity (inverse depth) using rounded alpha values.

        Returns:
            torch.Tensor: Mode depth map, shape [B, 1, H, W].
        """
        alpha_bin = self.alpha_bin
        mu_bin = (alpha_bin * self.mu).sum(1, True)
        sigma_bin = (alpha_bin * self.sigma).sum(1, True)
        return 1 / (mu_bin + 2 * sigma_bin.square())

    def mean_sigma(self):
        """Compute the weighted average of sigma parameters.

        Returns:
            torch.Tensor: Weighted sigma values, shape [B, 1, H, W].
        """
        return (self.alpha * self.sigma).sum(1, True)

    def diff_mu_rel(self):
        """Compute the relative difference between the two mu components.

        Returns the relative difference where 1 represents a 2x difference.

        Returns:
            torch.Tensor: Relative difference map, shape [B, 1, H, W].
        """
        # returns the relative difference between the two mu components where 1 is a 2 times difference
        r0 = self.mu[:, 1:2] / (self.mu[:, 0:1] + 1e-6)
        r1 = self.mu[:, 0:1] / (self.mu[:, 1:2] + 1e-6)
        return torch.maximum(r0, r1) - 1

    def diff_sigma(self):
        """Compute the log ratio of sigma parameters between components.

        Returns:
            torch.Tensor: Log ratio of sigma values, shape [B, 1, H, W].
        """
        return torch.log(self.sigma[:, 0:1] + 1e-6) - torch.log(self.sigma[:, 1:2] + 1e-6)

    def depth_likelihood(self, x, scale=1):
        """Compute the likelihood of depth values.

        Args:
            x (torch.Tensor): Bins values [N]
            scale (float or torch.Tensor): Scale factor.

        Returns:
            torch.Tensor: Likelihood values.
        """
        likelihood = torch.exp(-self.depth_nlog_likelihood(x, scale))  # [B, N, H, W]
        proba = likelihood / likelihood.sum(1, True)
        return proba

    def depth_nlog_likelihood(self, x, scale=1):
        """Compute the negative log likelihood of depth values.

        Args:
            x (torch.Tensor): Bins values [N]
            scale (float or torch.Tensor, optional): Scale factor. Defaults to 1.

        Returns:
            torch.Tensor: Negative log likelihood values. [B, N, H, W]
        """
        grid = x.view(1, 1, -1, 1, 1).expand(1, 1, -1, 192, 640)  # [1, 1, N, H, W]
        ll = self.nlog_likelihood_mix(
            self.transform(grid) * scale, self.alpha[:, :, None], self.mu[:, :, None], self.sigma[:, :, None]
        )
        return (ll - self.nlog_jacobian(grid, scale)).squeeze(1)  # [B, N, H, W]

    def likelihood(self, x):
        """Compute the likelihood of values in the distribution space.

        Args:
            x (torch.Tensor): Values in distribution space.

        Returns:
            torch.Tensor: Likelihood values.
        """
        likelihood = torch.exp(-self.nlog_likelihood(x))  # [B, N, H, W]
        proba = likelihood / likelihood.sum(1, True)
        return proba

    def nlog_likelihood(self, x):
        """Compute the negative log likelihood of values in distribution space.

        Args:
            x (torch.Tensor): Values in distribution space. [N]

        Returns:
            torch.Tensor: Negative log likelihood values. [B N H W]
        """
        grid = x.view(1, 1, -1, 1, 1).expand(1, 1, -1, 192, 640)
        return self.nlog_likelihood_mix(
            grid, self.alpha[:, :, None], self.mu[:, :, None], self.sigma[:, :, None]
        ).squeeze(1)

    @abstractmethod
    def nlog_likelihood_mix(self, x, alpha, mu, sigma):
        """Compute the negative log likelihood for mixture distribution.

        This is an abstract method that needs to be implemented by subclasses.

        Args:
            x (torch.Tensor): Values in distribution space.
            alpha (torch.Tensor): Mixture weights.
            mu (torch.Tensor): Mean parameters.
            sigma (torch.Tensor): Scale parameters.
        """
        pass

    def __getitem__(self, idx):
        """Get distribution for a specific batch index.

        Args:
            idx (int): Batch index.

        Returns:
            MixtureDistribution: Distribution for the specified batch index.
        """
        return self.__class__(self.mu[idx : idx + 1], self.sigma[idx : idx + 1], self.alpha[idx : idx + 1], self.inv)

    def __str__(self):
        """Return string representation of the distribution.

        Returns:
            str: String description of the distribution.
        """
        return (
            "Inverse Depth" if self.inv else "Depth"
        ) + f" {self.__class__.__name__} with {self.alpha.shape[1]} components"

    __repr__ = __str__


class GaussianMixture(MixtureDistribution):
    """Mixture of Gaussian distributions.

    This class implements a mixture of Gaussian distributions for modeling depth
    or inverse depth.
    """

    @staticmethod
    def nlog_likelihood_mix(x, alpha, mu, sigma):
        """Compute the negative log likelihood for a mixture of Gaussian distributions.

        Args:
            x (torch.Tensor): Values to evaluate, shape [..., 1, H, W].
            alpha (torch.Tensor): Mixture weights, shape [..., K, H, W].
            mu (torch.Tensor): Mean parameters, shape [..., K, H, W].
            sigma (torch.Tensor): Standard deviation parameters, shape [..., K, H, W].

        Returns:
            torch.Tensor: Negative log likelihood, shape [..., 1, H, W].
        """
        return -torch.logsumexp(
            -((x - mu) ** 2) / (2 * sigma**2 + 1e-6)
            + torch.log(alpha + 1e-6)
            - torch.log(sigma * sqrt(2 * torch.pi + 1e-6)),
            1,
            True,
        )

    def derivative(self, x):
        """Compute the derivative of the log likelihood with respect to x.

        Args:
            x (torch.Tensor): Input values.

        Returns:
            torch.Tensor: Derivative values.
        """
        diff = x - self.mu
        a = diff / (self.sigma**3 + 1e-6)
        exp = torch.exp(-(diff.square() / (2 * self.sigma.square())))
        return (self.alpha * a * exp).sum(1, True)

    def density(self, x):
        """Compute the probability density function at given depth values.

        Args:
            x (torch.Tensor): Depth values.

        Returns:
            torch.Tensor: Density values.
        """
        x = self.transform(x)
        return (self.alpha * gaussian(x, self.mu, self.sigma)).sum(1, True)

    def variance(self):
        """Compute the variance of the Gaussian mixture at each spatial location (K=2).

        Returns:
            torch.Tensor: Mixture variance map, shape [B, 1, H, W].
        """

        # Weighted mean
        mu_mix = (self.alpha * self.mu).sum(1, keepdim=True)  # [B, 1, H, W]
        # Variance for each component
        var_comp = self.sigma**2 + (self.mu - mu_mix) ** 2  # [B, 2, H, W]
        # Weighted sum
        var_mix = (self.alpha * var_comp).sum(1, keepdim=True)  # [B, 1, H, W]
        return var_mix

    def entropy_upper_bound(self):
        """Compute an upper bound on the entropy of the Gaussian mixture at each spatial location (K=2).

        Returns:
            torch.Tensor: Entropy upper bound map, shape [B, 1, H, W].
        """
        var_mix = self.variance()
        H_total = H_gaussian(torch.sqrt(var_mix + 1e-6))  # [B, 1, H, W]

        H_k = H_gaussian(self.sigma)  # [B, 2, H, W]
        H_w = -self.alpha * torch.log(self.alpha + 1e-6)  # [B, 2, H, W]
        H_sep = (H_w + self.alpha * H_k).sum(1, keepdim=True)  # [B, 1, H, W]

        return torch.minimum(H_total, H_sep)


def gaussian(x, mu, sigma):
    """Evaluate Gaussian probability density function.

    Args:
        x (torch.Tensor): Input values.
        mu (torch.Tensor): Mean parameters.
        sigma (torch.Tensor): Standard deviation parameters.

    Returns:
        torch.Tensor: Gaussian PDF values.
    """
    return torch.exp(-((x - mu) / sigma).square() / 2) / (sigma * sqrt(2 * torch.pi))


def H_gaussian(sigma):
    """Compute the entropy of a Gaussian distribution.

    Args:
        sigma (torch.Tensor): Standard deviation parameters.

    Returns:
        torch.Tensor: Entropy values.
    """
    return 0.5 * torch.log(2 * torch.pi * torch.e * sigma**2 + 1e-6)
