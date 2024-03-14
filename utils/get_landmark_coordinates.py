# This file contains the function to generate the center coordinates as tensor for the current net.
import torch


def landmark_coordinates(maps, grid_x=None, grid_y=None):
    """
    Generate the center coordinates as tensor for the current net.
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/lib.py#L19
    Parameters
    ----------
    maps: torch.Tensor
        Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    grid_x: torch.Tensor
        The grid x coordinates
    grid_y: torch.Tensor
        The grid y coordinates
    Returns
    ----------
    loc_x: Tensor
        The centroid x coordinates
    loc_y: Tensor
        The centroid y coordinates
    grid_x: Tensor
    grid_y: Tensor
    """
    return_grid = False
    if grid_x is None or grid_y is None:
        return_grid = True
        grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
                                        torch.arange(maps.shape[3]), indexing='ij')
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).contiguous().to(maps.device, non_blocking=True)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).contiguous().to(maps.device, non_blocking=True)
    map_sums = maps.sum(3).sum(2).detach()
    maps_x = grid_x * maps
    maps_y = grid_y * maps
    loc_x = maps_x.sum(3).sum(2) / map_sums
    loc_y = maps_y.sum(3).sum(2) / map_sums
    if return_grid:
        return loc_x, loc_y, grid_x, grid_y
    else:
        return loc_x, loc_y
