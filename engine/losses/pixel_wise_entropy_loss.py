# This file contains the pixel-wise entropy loss function
import torch


def pixel_wise_entropy_loss(maps):
    """
    Calculate pixel-wise entropy loss for a feature map
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return: value of the pixel-wise entropy loss
    """
    # Calculate entropy for each pixel with numerical stability
    entropy = torch.distributions.categorical.Categorical(probs=maps.permute(0, 2, 3, 1).contiguous()).entropy()
    # Take the mean of the entropy
    return entropy.mean()
