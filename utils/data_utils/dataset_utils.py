from PIL import Image
from torch import Tensor
from typing import List, Optional
import numpy as np
import torchvision
import json


def load_json(path: str):
    """
    Load json file from path and return the data
    :param path: Path to the json file
    :return:
    data: Data in the json file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data: dict, path: str):
    """
    Save data to a json file
    :param data: Data to be saved
    :param path: Path to save the data
    :return:
    """
    with open(path, "w") as f:
        json.dump(data, f)


def pil_loader(path):
    """
    Load image from path using PIL
    :param path: Path to the image
    :return:
    img: PIL Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_dimensions(image: Tensor):
    """
    Get the dimensions of the image
    :param image: Tensor or PIL Image or np.ndarray
    :return:
    h: Height of the image
    w: Width of the image
    """
    if isinstance(image, Tensor):
        _, h, w = image.shape
    elif isinstance(image, np.ndarray):
        h, w, _ = image.shape
    elif isinstance(image, Image.Image):
        w, h = image.size
    else:
        raise ValueError(f"Invalid image type: {type(image)}")
    return h, w


def center_crop_boxes_kps(img: Tensor, output_size: Optional[List[int]] = 448, parts: Optional[Tensor] = None,
                          boxes: Optional[Tensor] = None, num_keypoints: int = 15):
    """
    Calculate the center crop parameters for the bounding boxes and landmarks and update them
    :param img: Image
    :param output_size: Output size of the cropped image
    :param parts: Locations of the landmarks of following format: <part_id> <x> <y> <visible>
    :param boxes: Bounding boxes of the landmarks of following format: <image_id> <x> <y> <width> <height>
    :param num_keypoints: Number of keypoints
    :return:
    cropped_img: Center cropped image
    parts: Updated locations of the landmarks
    boxes: Updated bounding boxes of the landmarks
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        output_size = output_size
    else:
        raise ValueError(f"Invalid output size: {output_size}")

    crop_height, crop_width = output_size
    image_height, image_width = get_dimensions(img)
    img = torchvision.transforms.functional.center_crop(img, output_size)

    crop_top, crop_left = _get_center_crop_params_(image_height, image_width, output_size)

    if parts is not None:
        for j in range(num_keypoints):
            # Skip if part is invisible
            if parts[j][-1] == 0:
                continue
            parts[j][1] -= crop_left
            parts[j][2] -= crop_top

            # Skip if part is outside the crop
            if parts[j][1] > crop_width or parts[j][2] > crop_height:
                parts[j][-1] = 0
            if parts[j][1] < 0 or parts[j][2] < 0:
                parts[j][-1] = 0

            parts[j][1] = min(crop_width, parts[j][1])
            parts[j][2] = min(crop_height, parts[j][2])
            parts[j][1] = max(0, parts[j][1])
            parts[j][2] = max(0, parts[j][2])

    if boxes is not None:
        boxes[1] -= crop_left
        boxes[2] -= crop_top
        boxes[1] = max(0, boxes[1])
        boxes[2] = max(0, boxes[2])
        boxes[1] = min(crop_width, boxes[1])
        boxes[2] = min(crop_height, boxes[2])

    return img, parts, boxes


def _get_center_crop_params_(image_height: int, image_width: int, output_size: Optional[List[int]] = 448):
    """
    Get the parameters for center cropping the image
    :param image_height: Height of the image
    :param image_width: Width of the image
    :param output_size: Output size of the cropped image
    :return:
    crop_top: Top coordinate of the cropped image
    crop_left: Left coordinate of the cropped image
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        output_size = output_size
    else:
        raise ValueError(f"Invalid output size: {output_size}")

    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        crop_top, crop_left = padding_ltrb[1], padding_ltrb[0]
        return crop_top, crop_left

    if crop_width == image_width and crop_height == image_height:
        crop_top = 0
        crop_left = 0
        return crop_top, crop_left

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))

    return crop_top, crop_left
