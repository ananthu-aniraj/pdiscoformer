# Adapted from: https://pytorch.org/vision/stable/_modules/torchvision/datasets/flowers102.html#Flowers102
from torchvision import datasets
import PIL
from typing import Tuple, Any
import cv2
import numpy as np
import torch


class Flowers102Seg(datasets.Flowers102):
    """
    This class is a subclass of the torchvision.datasets.Flowers102 class that adds the segmentation images to the
    __getitem__ method.
    """

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, mask_transform=None):
        """
        Args:
        :param root:
        :param split:
        :param transform:
        :param target_transform:
        :param download:
        :param mask_transform: The transform to apply to the segmentation mask
        """
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self._seg_folder = self._base_folder / 'segmim'
        self.seg_files = []
        for image_file in self._image_files:
            image_name = image_file.name.split('_')[-1]
            seg_name = 'segmim_' + image_name
            seg_file = self._seg_folder / seg_name
            self.seg_files.append(seg_file)
        self.mask_transform = mask_transform
        self.num_classes = len(set(self._labels))

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")
        seg_image = PIL.Image.open(self.seg_files[idx]).convert("RGB")
        seg_image = np.array(seg_image)
        # Convert RGB to BGR
        seg_image = seg_image[:, :, ::-1].copy()
        # Convert to binary mask
        binary_mask = ((seg_image[:, :, 0] / (seg_image[:, :, 1] + seg_image[:, :, 2] + 1e-6)) > 100).astype(np.uint8)
        binary_mask = 1 - binary_mask
        if len(np.unique(binary_mask)) > 1:
            binary_mask = cv2.medianBlur(binary_mask, 5)
        seg_image = torch.as_tensor(binary_mask, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            seg_image = self.mask_transform(seg_image)

        if self.target_transform:
            label = self.target_transform(label)

        seg_image = seg_image.squeeze(0)

        return image, label, seg_image
