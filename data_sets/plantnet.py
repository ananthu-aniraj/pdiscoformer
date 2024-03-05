import torch
import os
import glob
from collections import defaultdict
from utils.data_utils.dataset_utils import pil_loader, load_json


class PlantNet(torch.utils.data.Dataset):
    """
    Class to train models on PlantNet300K
    Variables
        base_folder, str: Root directory of the dataset.
        image_sub_path, str: Path to the folder containing the images.
        transform, callable: A function/transform that takes in a PIL.Image and transforms it.

    """

    def __init__(self, base_folder, image_sub_path, transform=None, metadata_path=None, species_id_to_name_file=None):
        self.images_folder = os.path.join(base_folder, image_sub_path)

        self.transform = transform

        self.loader = pil_loader

        self.metadata = load_json(metadata_path)
        self.species_id_to_name = load_json(species_id_to_name_file)

        self.image_paths = glob.glob(os.path.join(self.images_folder, "**/*.jpg"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.jpeg"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.png"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.bmp"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.ppm"), recursive=True)
        self.image_paths += glob.glob(os.path.join(self.images_folder, "**/*.JPEG"), recursive=True)

        self.image_paths = sorted(self.image_paths)

        self.class_names = self.species_id_to_name.keys()
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        self.labels = [self.class_to_idx[os.path.basename(os.path.dirname(image_path))] for image_path in
                       self.image_paths]
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_names)
        self.idx_to_species_name = {idx: self.species_id_to_name[self.idx_to_class[idx]] for idx in
                                    range(self.num_classes)}
        self.per_class_count = defaultdict(int)
        self.class_to_img_ids = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.per_class_count[self.idx_to_class[label]] += 1
            self.class_to_img_ids[self.idx_to_class[label]].append(idx)
        # For top-K loss (class distribution)
        self.cls_num_list = [self.per_class_count[self.idx_to_class[idx]] for idx in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self.loader(image_path)
        label = self.labels[idx]
        image_name = image_path.split("/")[-1].split(".")[0]
        metadata = self.metadata[image_name]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, metadata

    def generate_class_balanced_indices(self, generator: torch.Generator, num_samples_per_class=10):
        indices = []
        for class_name, img_ids in self.class_to_img_ids.items():
            # randomly sample num_samples_per_class images from each class
            sampled_img_ids = torch.randperm(len(img_ids), generator=generator).tolist()
            if len(img_ids) > num_samples_per_class:
                sampled_img_ids = sampled_img_ids[:num_samples_per_class]
            indices.extend(sampled_img_ids)
        return indices


