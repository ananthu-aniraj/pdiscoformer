import torch
from utils.get_landmark_coordinates import landmark_coordinates


class ConcentrationLoss(torch.nn.Module):
    """
    This class defines the concentration loss.
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L15
    """

    def __init__(self):
        super(ConcentrationLoss, self).__init__()
        self.grid_x = None
        self.grid_y = None

    def forward(self, maps):
        """
        Forward function for the concentration loss.
        :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
        :return: The concentration loss
        """
        if self.grid_x is None or self.grid_y is None:
            grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
                                            torch.arange(maps.shape[3]), indexing='ij')
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).contiguous().to(maps.device, non_blocking=True)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).contiguous().to(maps.device, non_blocking=True)
            self.grid_x = grid_x
            self.grid_y = grid_y

        # Get landmark coordinates
        loc_x, loc_y = landmark_coordinates(maps, self.grid_x, self.grid_y)
        # Concentration loss
        loss_conc_x = ((loc_x.unsqueeze(-1).unsqueeze(-1).contiguous() - self.grid_x) / self.grid_x.shape[-1]) ** 2
        loss_conc_y = ((loc_y.unsqueeze(-1).unsqueeze(-1).contiguous() - self.grid_y) / self.grid_y.shape[-2]) ** 2
        loss_conc = (loss_conc_x + loss_conc_y) * maps
        return loss_conc[:, 0:-1, :, :].mean()
