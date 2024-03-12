import os
import torch.utils.data
from pycocotools.coco import COCO
import copy
import torch
import torch.utils.data
from collections import defaultdict
from utils.data_utils.dataset_utils import pil_loader
from .classes_mapping_imagenet import IMAGENET2012_CLASSES


class PartImageNetDataset(torch.utils.data.Dataset):
    """PartImageNet dataset"""

    def __init__(self, data_path: str, transform=None,
                 get_masks=False, image_sub_path='train', annotation_file_path="train.json", class_names_to_idx=None,
                 class_idx_to_names=None, class_names=None, mask_transform=None):
        """
        Args:
            data_path (string): path to the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
            get_masks (bool): whether to return the masks along with the images
            image_sub_path (str): sub path to the images
            annotation_file_path (str): path to the annotation file
            class_names_to_idx (dict): dictionary mapping class names to indices
            class_idx_to_names (dict): dictionary mapping class indices to names
            class_names (list): list of class names
            mask_transform (callable, optional): Optional transform to be applied
                on the masks.
        """
        self.data_path = data_path
        self.transform = transform
        self.get_masks = get_masks
        self.loader = pil_loader
        self.image_sub_path = image_sub_path
        self.coco = COCO(annotation_file_path)
        self._preprocess_annotations()
        self.image_ids = [img_dict['id'] for img_dict in self.coco.imgs.values()]
        # Number of key-points in the dataset (Ground truth parts)
        self.num_kps = len(self.coco.cats)
        # Coarse-grained classes in the dataset
        self.super_categories = list(dict.fromkeys([self.coco.cats[cat]['supercategory'] for cat in self.coco.cats]))
        self.super_categories.sort()
        self.mask_transform = mask_transform
        self.img_id_to_label = {}
        self.image_id_to_name = {}
        self.img_id_to_supercat = {}

        if class_names is None and class_names_to_idx is None and class_idx_to_names is None:
            self.class_names = []

            for img_dict in self.coco.imgs.values():
                img_name = os.path.basename(img_dict['file_name'])
                class_name_wordnet = img_name.split('_')[0]
                self.class_names.append(IMAGENET2012_CLASSES[class_name_wordnet])

            self.class_names = list(dict.fromkeys(self.class_names))
            self.class_names.sort()
            self.num_classes = len(self.class_names)
            self.class_names_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
            self.class_idx_to_names = {idx: class_name for idx, class_name in enumerate(self.class_names)}
        else:
            self.class_names_to_idx = class_names_to_idx
            self.class_idx_to_names = class_idx_to_names
            self.class_names = class_names
            self.num_classes = len(self.class_names)
        filtered_img_iterator = 0
        self.filtered_img_id_to_orig_img_id = {}
        self.img_ids_filtered = []
        # Number of instances per class
        self.per_class_count = defaultdict(int)
        for image_id in self.image_ids:
            annIds = self.coco.getAnnIds(imgIds=image_id, iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            img_name = self.coco.loadImgs(image_id)[0]['file_name']

            if anns:
                cats = [ann['category_id'] for ann in anns if ann['area'] > 0]
                supercat_img = list(dict.fromkeys([self.coco.cats[cat]['supercategory'] for cat in cats]))[0]
                class_name_wordnet = os.path.basename(img_name).split('_')[0]
                class_idx = self.class_names_to_idx[IMAGENET2012_CLASSES[class_name_wordnet]]
                self.image_id_to_name[filtered_img_iterator] = os.path.join(self.data_path, self.image_sub_path,
                                                                            img_name)
                self.img_ids_filtered.append(filtered_img_iterator)
                self.img_id_to_label[filtered_img_iterator] = class_idx
                self.filtered_img_id_to_orig_img_id[filtered_img_iterator] = image_id
                self.img_id_to_supercat[filtered_img_iterator] = supercat_img
                self.per_class_count[self.class_idx_to_names[class_idx]] += 1
                filtered_img_iterator += 1
        # For top-K loss (class distribution)
        self.cls_num_list = [self.per_class_count[self.class_idx_to_names[idx]] for idx in range(self.num_classes)]

    def __len__(self):
        return len(self.img_ids_filtered)

    def _preprocess_annotations(self):
        json_dict = copy.deepcopy(self.coco.dataset)
        for ann in json_dict['annotations']:
            if ann["area"] == 0 or ann["iscrowd"] == 1:
                continue
            for poly_num, seg in enumerate(ann['segmentation']):
                if len(seg) == 4:
                    x1, y1, w, h = ann['bbox']
                    x2 = x1 + w
                    y2 = y1 + h
                    seg_poly = [x1, y1, x1, y2, x2, y2, x2, y1]
                    ann['segmentation'][poly_num] = seg_poly
        self.coco.dataset = copy.deepcopy(json_dict)
        self.coco.createIndex()

    def __getitem__(self, idx):
        img_id = self.img_ids_filtered[idx]
        img_path = self.image_id_to_name[img_id]
        im = self.loader(img_path)
        label = self.img_id_to_label[img_id]
        if self.transform:
            im = self.transform(im)
        if not self.get_masks:
            return im, label
        mask = self.getmasks(img_id)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return im, label, mask

    def getmasks(self, img_id):
        coco = self.coco
        original_img_id = self.filtered_img_id_to_orig_img_id[img_id]
        anns = coco.imgToAnns[original_img_id]
        img = coco.imgs[original_img_id]
        mask_tensor = torch.zeros(size=(self.num_kps, img['height'], img['width']))
        for i, ann in enumerate(anns):
            if ann["area"] == 0 or ann["iscrowd"] == 1:
                continue
            cat = ann['category_id']
            mask = torch.as_tensor(coco.annToMask(ann), dtype=torch.float32)
            mask_tensor[cat] += mask
        return mask_tensor

    def generate_supercat_subset_all(self):
        supercat_to_img_ids = defaultdict(list)
        for img_id in self.img_ids_filtered:
            supercat_to_img_ids[self.img_id_to_supercat[img_id]].append(img_id)
        return supercat_to_img_ids


if __name__ == '__main__':
    pass
