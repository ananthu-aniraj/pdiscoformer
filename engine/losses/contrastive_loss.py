import torch
from torch import Tensor


def contrastive_loss(dist: Tensor, eps: float = 1e-10):
    """
    Calculate contrastive loss for a feature map with shape (batch_size, num_parts, *feature_map_shape)
    :param dist: Matrix of dissimilarities between the feature vectors
    :param eps: Numerical stability parameter
    :return:
    """
    dist = torch.sqrt(dist)
    loss = dist.min(dim=1).values / (dist.mean(1) + eps)  # add epsilon to prevent NaNs
    loss = torch.nn.functional.binary_cross_entropy(loss, torch.zeros_like(loss))
    return loss
