import torch
import os
import glob
from collections import defaultdict
from .classes_mapping_imagenet import imagenet_idx_to_class_names, imagenet_class_names_to_idx, IMAGENET2012_CLASSES
from utils.data_utils.dataset_utils import pil_loader


class ImageNetWithOODEval(torch.utils.data.Dataset):
    """
    Class to train models on ImageNet with Eval on OOD sets
    Variables
        base_folder, str: Root directory of the dataset.
        image_sub_path, str: Path to the folder containing the images.
        transform, callable: A function/transform that takes in a PIL.Image and transforms it.
    """

    def __init__(self, base_folder, image_sub_path, transform=None):
        self.class_to_idx = imagenet_class_names_to_idx
        self.idx_to_class = imagenet_idx_to_class_names

        self.images_folder = os.path.join(base_folder, image_sub_path)

        self.num_classes = len(imagenet_idx_to_class_names)

        self.classes = list(self.class_to_idx.keys())

        self.wordnet_to_class_name = IMAGENET2012_CLASSES

        self.transform = transform

        self.loader = pil_loader

        self.image_paths = glob.glob(os.path.join(self.images_folder, "**/*.jpg"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.jpeg"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.png"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.bmp"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.ppm"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.JPEG"), recursive=True)

        self.image_paths = sorted(self.image_paths)
        self.per_class_count = defaultdict(int)
        self.labels = [self.class_to_idx[self.wordnet_to_class_name[os.path.basename(os.path.dirname(image_path))]] for
                       image_path in self.image_paths]
        for label in self.labels:
            self.per_class_count[self.idx_to_class[label]] += 1
        self.cls_num_list = [self.per_class_count[self.idx_to_class[idx]] for idx in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.loader(image_path)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

