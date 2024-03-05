"""Adapted from: https://github.com/zxhuang1698/interpretability-by-parts/"""

import torch
import torch.utils.data as data
import os
import os.path
import pickle
import numpy as np
from collections import defaultdict
from utils.data_utils.dataset_utils import pil_loader


class CelebA(data.Dataset):
    """
    CelebA dataset.
    Variables
    ----------
        root, str: Root directory of the dataset.
        split, str: Current data split.
            "train": Training split without MAFL images. (For localization)
            "train_full": Training split with MAFL images. (For classification)
            "val": Validation split for classification accuracy.
            "test": Testing split for classification accuracy.
            "fit": Split for fitting the linear regressor.
            "eval": Split for evaluating the linear regressor.
        align, bool: Whether use aligned version or not.
        percentage, float: For unaligned version, the least percentage of (face area / image area)
        transform, callable: A function/transform that takes in a PIL.Image and transforms it.
        resize, tuple: The size of image (h, w) after transformation (This version does not support cropping)
    """

    def __init__(self, root,
                 split='train', align=False,
                 percentage=None, transform=None, resize=256, image_sub_path="unaligned"):

        self.root = root
        self.split = split
        self.align = align
        self.resize = resize
        self.image_sub_path = image_sub_path
        # load the dictionary for data
        align_name = '_aligned' if align else '_unaligned'
        percentage_name = '_0' if percentage is None else '_' + str(int(percentage * 100))
        save_name = os.path.join(root, split + align_name + percentage_name + '.pickle')
        self.shuffle = np.arange(182637)
        np.random.shuffle(self.shuffle)
        if os.path.exists(save_name) is False:
            print('Preparing the data...')
            self.generate_dict(save_name)
            print('Data dictionary created and saved.')
        with open(save_name, 'rb') as handle:
            save_dict = pickle.load(handle)

        self.images = save_dict['images']  # image filenames
        self.landmarks = save_dict['landmarks']  # 5 face landmarks
        self.targets = save_dict['targets']  # binary labels
        self.bboxes = save_dict['bboxes']  # x y w h
        self.sizes = save_dict['sizes']  # height width
        self.identities = save_dict['identities']
        self.transform = transform
        self.loader = pil_loader

        # select a subset of the current data split according the face area
        if percentage is not None:
            new_images = []
            new_landmarks = []
            new_targets = []
            new_bboxes = []
            new_sizes = []
            new_identities = []
            for i in range(len(self.images)):
                if float(self.bboxes[i][-1] * self.bboxes[i][-2]) >= float(
                        self.sizes[i][-1] * self.sizes[i][-2]) * percentage:
                    new_images.append(self.images[i])
                    new_landmarks.append(self.landmarks[i])
                    new_targets.append(self.targets[i])
                    new_bboxes.append(self.bboxes[i])
                    new_sizes.append(self.sizes[i])
                    new_identities.append(self.identities[i])
            self.images = new_images
            self.landmarks = new_landmarks
            self.targets = new_targets
            self.bboxes = new_bboxes
            self.sizes = new_sizes
            self.identities = new_identities
        print('Number of classes in the ' + self.split + ' split: ' + str(max(self.identities)))
        print('Number of samples in the ' + self.split + ' split: ' + str(len(self.images)))
        self.num_classes = max(self.identities)
        self.per_class_count = defaultdict(int)
        for label in self.identities:
            self.per_class_count[label] += 1
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]

    # generate a dictionary for a certain data split
    def generate_dict(self, save_name):

        print('Start generating data dictionary as ' + save_name)

        full_img_list = []
        ann_file = 'list_attr_celeba.txt'
        bbox_file = 'list_bbox_celeba.txt'
        size_file = 'list_imsize_celeba.txt'
        identity_file = 'identity_CelebA.txt'

        if self.align is True:
            landmark_file = 'list_landmarks_align_celeba.txt'
        else:
            landmark_file = 'list_landmarks_unalign_celeba.txt'

        # load all the images according to the current split
        if self.split == 'train':
            imgfile = 'celebA_training.txt'
        elif self.split == 'val':
            imgfile = 'celebA_validating.txt'
        elif self.split == 'test':
            imgfile = 'celebA_testing.txt'
        elif self.split == 'fit':
            imgfile = 'MAFL_training.txt'
        elif self.split == 'eval':
            imgfile = 'MAFL_testing.txt'
        elif self.split == 'train_full':
            imgfile = 'celebA_training_full.txt'
        for line in open(os.path.join(self.root, imgfile), 'r'):
            full_img_list.append(line.split()[0])

        # prepare the indexes and convert annotation files to lists
        full_img_list_idx = [(int(s.rstrip(".jpg")) - 1) for s in full_img_list]
        ann_full_list = [line.split() for line in open(os.path.join(self.root, ann_file), 'r')]
        bbox_full_list = [line.split() for line in open(os.path.join(self.root, bbox_file), 'r')]
        size_full_list = [line.split() for line in open(os.path.join(self.root, size_file), 'r')]
        landmark_full_list = [line.split() for line in open(os.path.join(self.root, landmark_file), 'r')]
        identity_full_list = [line.split() for line in open(os.path.join(self.root, identity_file), 'r')]

        # assertion
        assert len(ann_full_list[0]) == 41
        assert len(bbox_full_list[0]) == 5
        assert len(size_full_list[0]) == 3
        assert len(landmark_full_list[0]) == 11

        # select samples and annotations for the current data split
        # init the lists
        filename_list = []
        target_list = []
        landmark_list = []
        bbox_list = []
        size_list = []
        identity_list = []

        # select samples and annotations
        for i in full_img_list_idx:
            idx = self.shuffle[i]

            # assertion
            assert (idx + 1) == int(ann_full_list[idx][0].rstrip(".jpg"))
            assert (idx + 1) == int(bbox_full_list[idx][0].rstrip(".jpg"))
            assert (idx + 1) == int(size_full_list[idx][0].rstrip(".jpg"))
            assert (idx + 1) == int(landmark_full_list[idx][0].rstrip(".jpg"))

            # append the filenames and annotations
            filename_list.append(ann_full_list[idx][0])
            target_list.append([int(i) for i in ann_full_list[idx][1:]])
            bbox_list.append([int(i) for i in bbox_full_list[idx][1:]])
            size_list.append([int(i) for i in size_full_list[idx][1:]])
            landmark_list_xy = []
            for j in range(5):
                landmark_list_xy.append(
                    [int(landmark_full_list[idx][1 + 2 * j]), int(landmark_full_list[idx][2 + 2 * j])])
            landmark_list.append(landmark_list_xy)
            identity_list.append(int(identity_full_list[idx][1]))

        # expand the filename to the full path
        full_path_list = [os.path.join(self.root, self.image_sub_path, filename) for filename in filename_list]

        # create the dictionary and save it on the disk
        save_dict = dict()
        save_dict['images'] = full_path_list
        save_dict['landmarks'] = landmark_list
        save_dict['targets'] = target_list
        save_dict['bboxes'] = bbox_list
        save_dict['sizes'] = size_list
        save_dict['identities'] = identity_list
        with open(save_name, 'wb') as handle:
            pickle.dump(save_dict, handle)

    def __getitem__(self, index):
        """
        Retrieve data samples.
        Args
        ----------
        index: int
            Index of the sample.
        Returns
        ----------
        sample: PIL.Image
            Image of the given index.
        identity: torch.LongTensor
            Corresponding identity labels for all images
        landmark_locs: torch.FloatTensor, [5, 2]
            Landmark annotations, column first.
        """
        # load images and targets
        path = self.images[index]
        sample = self.loader(path)
        identity = self.identities[index] - 1
        image = np.array(sample)
        if image.shape[-3] > image.shape[-2]:
            factor = self.resize / image.shape[-2]
        else:
            factor = self.resize / image.shape[-3]

        # transform the image and target
        if self.transform is not None:
            sample = self.transform(sample)

        # processing the landmarks
        landmark_locs = self.landmarks[index]
        landmark_locs = torch.LongTensor(landmark_locs).float()
        landmark_locs[:, 0] = landmark_locs[:, 0] * factor
        landmark_locs[:, 1] = landmark_locs[:, 1] * factor
        return sample, identity, landmark_locs

    def __len__(self):
        return len(self.images)
