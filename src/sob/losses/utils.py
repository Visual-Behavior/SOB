def cross_entropy(x, y):
    """Compute cross entropy loss.

    Args:
        x (torch.Tensor): Predicted values.
        y (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Cross entropy loss.
    """
    return -y * (x + 1e-6).log()


def mse(x, y):
    """Compute mean squared error loss with scaling.

    Args:
        x (torch.Tensor): Predicted values.
        y (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Scaled mean squared error loss.
    """
    return 1e-2 * (x - y).square()


def mae(x, y):
    """Compute mean absolute error loss with scaling.

    Args:
        x (torch.Tensor): Predicted values.
        y (torch.Tensor): Target values.

    Returns:
        torch.Tensor: Scaled mean absolute error loss.
    """
    return 1e-2 * (x - y).abs()


def attn(x, y):
    """Compute attention-weighted loss.

    Weight predictions by inverse of target values,
    giving more attention to areas where target is low.

    Args:
        x (torch.Tensor): Predicted values.
        y (torch.Tensor): Target values/weights.

    Returns:
        torch.Tensor: Attention-weighted values.
    """
    return x * (1 - y)
