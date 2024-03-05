"""Class CUB200 from: https://github.com/zxhuang1698/interpretability-by-parts/"""
import os
import pandas as pd
import PIL.Image
from skimage.io import imread
import torch.utils.data
import numpy as np
import PIL.Image
import torch
import torch.utils.data
from collections import defaultdict
from utils.data_utils.dataset_utils import pil_loader, _center_crop_boxes_kps_


class CUBDataset(torch.utils.data.Dataset):
    """
    CUB200-2011 dataset.
    Variables
    ----------
        data_path, str: Root directory of the dataset.
        split, int: Percentage of training samples to use for training.
        mode, str: Current data split.
            "train": Training split
            "val": Validation split
            "test": Testing split
        height, int: Height of the images
        transform, callable: A function/transform that takes in a PIL.Image and transforms it.
        train_samples, list: List of training samples to use for validation
        image_sub_path, str: Path to the folder containing the images.
    """
    def __init__(self, data_path, split=1, mode='train', height: int = 256,
                 transform=None, train_samples=None, image_sub_path="images"):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.image_sub_path = image_sub_path
        train_test = pd.read_csv(os.path.join(data_path, 'train_test_split.txt'), delim_whitespace=True,
                                 names=['id', 'train'])
        image_names = pd.read_csv(os.path.join(data_path, 'images.txt'), delim_whitespace=True,
                                  names=['id', 'filename'])
        labels = pd.read_csv(os.path.join(data_path, 'image_class_labels.txt'), delim_whitespace=True,
                             names=['id', 'label'])
        image_parts = pd.read_csv(os.path.join(data_path, 'parts', 'part_locs.txt'), delim_whitespace=True,
                                  names=['id', 'part_id', 'x', 'y', 'visible'])
        dataset = train_test.merge(image_names, on='id')
        dataset = dataset.merge(labels, on='id')

        if mode == 'train':
            dataset = dataset.loc[dataset['train'] == 1]
            samples = np.arange(len(dataset))
            np.random.shuffle(samples)
            self.trainsamples = samples[:int(len(samples) * split)]
            dataset = dataset.iloc[self.trainsamples]
        elif mode == 'test':
            dataset = dataset.loc[dataset['train'] == 0]
        elif mode == 'val':
            dataset = dataset.loc[dataset['train'] == 1]
            if train_samples is None:
                raise RuntimeError('Please provide the list of training samples'
                                   'to the validation dataset')
            dataset = dataset.drop(dataset.index[train_samples])

        # training images are labelled 1, test images labelled 0. Add these
        # images to the list of image IDs
        self.ids = np.array(dataset['id'])
        self.names = np.array(dataset['filename'])
        # Subtract 1 because classes run from 1-200 instead of 0-199
        self.labels = np.array(dataset['label']) - 1
        parts = {}
        for i in self.ids:
            parts[i] = image_parts[image_parts['id'] == i]
        self.parts = parts
        self.height = height
        self.num_classes = len(np.unique(self.labels))
        self.per_class_count = defaultdict(int)
        for label in self.labels:
            self.per_class_count[label] += 1
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.image_sub_path, self.names[idx])
        im = imread(image_path)
        label = self.labels[idx]

        if len(im.shape) == 2:
            im = np.stack((im,) * 3, axis=-1)

        if self.transform:
            im_orig = PIL.Image.fromarray(im)
            im = self.transform(im_orig)

        return im, label

    def get_visible_parts(self, idx):
        """
        Returns all parts that are visible in the current image
        Parameters
        ----------
        idx: int
            The index for which to retrieve the visible parts
        """
        dataset_id = self.ids[idx]
        parts = self.parts[dataset_id]
        return parts


class CUB200(torch.utils.data.Dataset):
    """
    From: https://github.com/zxhuang1698/interpretability-by-parts/
    CUB200 dataset.
    Variables
    ----------
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _train_data, list of np.array.
        _train_labels, list of int.
        _train_parts, list np.array.
        _train_boxes, list np.array.
        _test_data, list of np.array.
        _test_labels, list of int.
        _test_parts, list np.array.
        _test_boxes, list np.array.
    """

    def __init__(self, root, train=True, transform=None, resize=448, center_crop=False, image_sub_path="images"):
        """
        Load the dataset.
        Args
        ----------
        root: str
            Root directory of the dataset.
        train: bool
            train/test data split.
        transform: callable
            A function/transform that takes in a PIL.Image and transforms it.
        resize: int
            Length of the shortest of edge of the resized image. Used for transforming landmarks and bounding boxes.
        """
        self._root = root
        self._train = train
        self._transform = transform
        self.loader = pil_loader
        self.newsize = resize
        self.center_crop = center_crop
        self.image_sub_path = image_sub_path
        # 15 key points provided by CUB
        self.num_kps = 15

        if not os.path.isdir(root):
            os.mkdir(root)

        self.per_class_count = defaultdict(int)
        # Load all data into memory for best IO efficiency. This might take a while
        if self._train:
            self._train_data, self._train_labels, self._train_parts, self._train_boxes = self._get_file_list(train=True)
            assert (len(self._train_data) == 5994
                    and len(self._train_labels) == 5994)
            self.num_classes = len(np.unique(self._train_labels))
            for label in self._train_labels:
                self.per_class_count[label] += 1
        else:
            self._test_data, self._test_labels, self._test_parts, self._test_boxes = self._get_file_list(train=False)
            assert (len(self._test_data) == 5794
                    and len(self._test_labels) == 5794)
            self.num_classes = len(np.unique(self._test_labels))
            for label in self._test_labels:
                self.per_class_count[label] += 1

        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]

    def __getitem__(self, index):
        """
        Retrieve data samples.
        Args
        ----------
        index: int
            Index of the sample.
        Returns
        ----------
        image: torch.FloatTensor, [3, H, W]
            Image of the given index.
        target: int
            Label of the given index.
        parts: torch.FloatTensor, [15, 4]
            Landmark annotations.
        boxes: torch.FloatTensor, [5, ]
            Bounding box annotations.
        """
        # load the variables according to the current index and split
        if self._train:
            image_path = self._train_data[index]
            target = self._train_labels[index]
            parts = self._train_parts[index]
            boxes = self._train_boxes[index]

        else:
            image_path = self._test_data[index]
            target = self._test_labels[index]
            parts = self._test_parts[index]
            boxes = self._test_boxes[index]

        # load the image
        image = self.loader(image_path)
        image = np.array(image)

        # numpy arrays to pytorch tensors
        parts = torch.from_numpy(parts).float()
        boxes = torch.from_numpy(boxes).float()

        # calculate the resize factor
        # if original image height is larger than width, the real resize factor is based on width
        if image.shape[0] >= image.shape[1]:
            factor = self.newsize / image.shape[1]
        else:
            factor = self.newsize / image.shape[0]

        # transform 15 landmarks according to the new shape
        # each landmark has a 4-element annotation: <landmark_id, column, row, existence>
        for j in range(self.num_kps):

            # step in only when the current landmark exists
            if abs(parts[j][-1]) > 1e-5:
                # calculate the new location according to the new shape
                parts[j][-3] = parts[j][-3] * factor
                parts[j][-2] = parts[j][-2] * factor

        # rescale the annotation of bounding boxes
        # the annotation format of the bounding boxes are <image_id, col of top-left corner, row of top-left corner, width, height>
        boxes[1:] *= factor

        # convert the image into a PIL image for transformation
        image = PIL.Image.fromarray(image)

        # apply transformation
        if self._transform is not None:
            image = self._transform(image)

        # center crop
        if self.center_crop:
            image, parts, boxes = _center_crop_boxes_kps_(image, self.newsize, parts, boxes, self.num_kps)
        return image, target, parts, boxes, image_path

    def __len__(self):
        """Return the length of the dataset."""
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _get_file_list(self, train=True):
        """Prepare the data for train/test split and save onto disk."""

        # load the list into numpy arrays
        image_path = os.path.join(self._root, self.image_sub_path)
        id2name = np.genfromtxt(os.path.join(self._root, 'images.txt'), dtype=str)
        id2train = np.genfromtxt(os.path.join(self._root, 'train_test_split.txt'), dtype=int)
        id2part = np.genfromtxt(os.path.join(self._root, 'parts', 'part_locs.txt'), dtype=float)
        id2box = np.genfromtxt(os.path.join(self._root, 'bounding_boxes.txt'), dtype=float)

        # creat empty lists
        train_data = []
        train_labels = []
        train_parts = []
        train_boxes = []
        test_data = []
        test_labels = []
        test_parts = []
        test_boxes = []

        # iterating all samples in the whole dataset
        for id_ in range(id2name.shape[0]):
            # load each variable
            image = os.path.join(image_path, id2name[id_, 1])
            # Label starts with 0
            label = int(id2name[id_, 1][:3]) - 1
            parts = id2part[id_ * self.num_kps: id_ * self.num_kps + self.num_kps][:, 1:]
            boxes = id2box[id_]

            # training split
            if id2train[id_, 1] == 1:
                train_data.append(image)
                train_labels.append(label)
                train_parts.append(parts)
                train_boxes.append(boxes)
            # testing split
            else:
                test_data.append(image)
                test_labels.append(label)
                test_parts.append(parts)
                test_boxes.append(boxes)

        # return according to different splits
        if train:
            return train_data, train_labels, train_parts, train_boxes
        else:
            return test_data, test_labels, test_parts, test_boxes


if __name__ == '__main__':
    pass
