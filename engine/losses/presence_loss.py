import torch


def presence_loss_soft_constraint(maps: torch.Tensor, beta: float = 0.1):
    """
    Calculate presence loss for a feature map
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :param beta: Weight of soft constraint
    :return: value of the presence loss
    """
    loss_max = torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1).max(dim=0)[0]
    loss_max_detach = loss_max.detach().clone()
    loss_max_p1 = 1 - loss_max
    loss_max_p2 = ((1 - beta) * loss_max_detach) + beta
    loss_max_final = (loss_max_p1 * loss_max_p2).mean()
    return loss_max_final


def presence_loss_tanh(maps: torch.Tensor):
    """
    Calculate presence loss for a feature map with tanh formulation from the paper PIP-NET
    Ref: https://github.com/M-Nauta/PIPNet/blob/68054822ee405b5f292369ca846a9c6233f2df69/pipnet/train.py#L111
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return:
    """
    pooled_maps = torch.tanh(torch.sum(torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1), dim=0))

    loss_max = torch.nn.functional.binary_cross_entropy(pooled_maps, target=torch.ones_like(pooled_maps))

    return loss_max


def presence_loss_soft_tanh(maps: torch.Tensor):
    """
    Calculate presence loss for a feature map with tanh formulation (non-log/softer version)
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return:
    """
    pooled_maps = torch.tanh(torch.sum(torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1), dim=0))

    loss_max = 1 - pooled_maps

    return loss_max.mean()


def presence_loss_original(maps: torch.Tensor):
    """
    Calculate presence loss for a feature map
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L181
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return: value of the presence loss
    """

    loss_max = torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1).max(dim=0)[0].mean()

    return 1 - loss_max


class PresenceLoss(torch.nn.Module):
    """
    This class defines the presence loss.
    """

    def __init__(self, loss_type: str = "original", beta: float = 0.1):
        super(PresenceLoss, self).__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(self, maps):
        """
        Forward function for the presence loss.
        :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
        :return: The presence loss
        """
        if self.loss_type == "original":
            return presence_loss_original(maps)
        elif self.loss_type == "soft_constraint":
            return presence_loss_soft_constraint(maps, beta=self.beta)
        elif self.loss_type == "tanh":
            return presence_loss_tanh(maps)
        elif self.loss_type == "soft_tanh":
            return presence_loss_soft_tanh(maps)
        else:
            raise NotImplementedError(f"Presence loss {self.loss_type} not implemented")
