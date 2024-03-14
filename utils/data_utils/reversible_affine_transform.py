# Description: This file contains the code for the reversible affine transform
import torchvision.transforms as transforms
import torch
from typing import List, Optional, Tuple, Any


def generate_affine_trans_params(
        degrees: List[float],
        translate: Optional[List[float]],
        scale_ranges: Optional[List[float]],
        shears: Optional[List[float]],
        img_size: List[int],
) -> Tuple[float, Tuple[int, int], float, Any]:
    """Get parameters for affine transformation

    Returns:
        params to be passed to the affine transformation
    """
    angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
    if translate is not None:
        max_dx = float(translate[0] * img_size[0])
        max_dy = float(translate[1] * img_size[1])
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        translations = (tx, ty)
    else:
        translations = (0, 0)

    if scale_ranges is not None:
        scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
    else:
        scale = 1.0

    shear_x = shear_y = 0.0
    if shears is not None:
        shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
        if len(shears) == 4:
            shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

    shear = (shear_x, shear_y)
    if shear_x == 0.0 and shear_y == 0.0:
        shear = 0.0

    return angle, translations, scale, shear


def rigid_transform(img, angle, translate, scale, invert=False, shear=0,
                    interpolation=transforms.InterpolationMode.BILINEAR):
    """
    Affine transforms input image
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/lib.py#L54
    Parameters
    ----------
    img: Tensor
        Input image
    angle: int
        Rotation angle between -180 and 180 degrees
    translate: [int]
        Sequence of horizontal/vertical translations
    scale: float
        How to scale the image
    invert: bool
        Whether to invert the transformation
    shear: float
        Shear angle in degrees
    interpolation: InterpolationMode
        Interpolation mode to calculate output values
    Returns
    ----------
    img: Tensor
        Transformed image

    """
    if not invert:
        img = transforms.functional.affine(img, angle=angle, translate=translate, scale=scale, shear=shear,
                                           interpolation=interpolation)
    else:
        translate = [-t for t in translate]
        img = transforms.functional.affine(img=img, angle=0, translate=translate, scale=1, shear=shear)
        img = transforms.functional.affine(img=img, angle=-angle, translate=[0, 0], scale=1 / scale, shear=shear)

    return img
