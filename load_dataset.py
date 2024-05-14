import torch
import os
from data_sets import FineGrainedBirdClassificationDataset, CelebA, ImageNetWithOODEval, PartImageNetDataset, PlantNet
from torchvision import datasets
from collections import Counter


def get_dataset(args, train_transforms, test_transforms):
    if args.dataset == 'cub' or args.dataset == 'nabirds':
        dataset_train = FineGrainedBirdClassificationDataset(args.data_path, split=args.train_split, mode='train',
                                                             transform=train_transforms,
                                                             image_sub_path=args.image_sub_path_train)
        dataset_test = FineGrainedBirdClassificationDataset(args.data_path, mode=args.eval_mode,
                                                            transform=test_transforms,
                                                            image_sub_path=args.image_sub_path_test)
        num_cls = dataset_train.num_classes
    elif args.dataset == 'celeba':
        dataset_train = CelebA(args.data_path, split='train', align=False, percentage=0.3,
                               transform=train_transforms, resize=args.image_size,
                               image_sub_path=args.image_sub_path_train)
        dataset_test = CelebA(args.data_path, split=args.eval_mode, align=False, percentage=0.3,
                              transform=test_transforms, resize=args.image_size,
                              image_sub_path=args.image_sub_path_test)
        num_cls = dataset_train.num_classes
    elif args.dataset == 'pug':
        if args.eval_mode == 'val' or args.eval_mode == 'train':
            dataset_train = datasets.ImageFolder(os.path.join(args.data_path, args.image_sub_path_train),
                                                 train_transforms)
            train_class_to_num_instances = Counter(dataset_train.targets)
            dataset_train.cls_num_list = [train_class_to_num_instances[idx] for idx in
                                          range(len(dataset_train.classes))]
            dataset_test = datasets.ImageFolder(os.path.join(args.data_path, args.image_sub_path_test), test_transforms)
        else:
            dataset_train_p1 = datasets.ImageFolder(os.path.join(args.data_path, args.image_sub_path_train),
                                                    train_transforms)
            p1_class_to_num_instances = Counter(dataset_train_p1.targets)
            dataset_train_p2 = datasets.ImageFolder(os.path.join(args.data_path, 'val'), train_transforms)
            p2_class_to_num_instances = Counter(dataset_train_p2.targets)
            train_class_to_num_instances = p1_class_to_num_instances + p2_class_to_num_instances
            dataset_train = torch.utils.data.ConcatDataset([dataset_train_p1, dataset_train_p2])
            dataset_train.cls_num_list = [train_class_to_num_instances[idx] for idx in
                                          range(len(dataset_train.classes))]
            dataset_test = datasets.ImageFolder(os.path.join(args.data_path, args.image_sub_path_test), test_transforms)
        test_class_to_num_instances = Counter(dataset_test.targets)
        dataset_test.cls_num_list = [test_class_to_num_instances[idx] for idx in range(len(dataset_test.classes))]
        num_cls = len(dataset_test.classes)
    elif args.dataset == 'imagenet':
        if args.eval_mode == 'val' or args.eval_mode == 'train':
            dataset_train = ImageNetWithOODEval(args.data_path, args.image_sub_path_train,
                                                transform=train_transforms)
            dataset_test = ImageNetWithOODEval(args.data_path, args.image_sub_path_test,
                                               transform=test_transforms)
        else:
            dataset_train_p1 = ImageNetWithOODEval(args.data_path, args.image_sub_path_train,
                                                   transform=train_transforms)
            p1_cls_num_list = dataset_train_p1.cls_num_list
            dataset_train_p2 = ImageNetWithOODEval(args.data_path, 'val',
                                                   transform=train_transforms)
            p2_cls_num_list = dataset_train_p2.cls_num_list
            train_cls_num_list = [p1_cls_num_list[idx] + p2_cls_num_list[idx] for idx in
                                  range(len(p1_cls_num_list))]
            dataset_train = torch.utils.data.ConcatDataset([dataset_train_p1, dataset_train_p2])
            dataset_train.cls_num_list = train_cls_num_list
            dataset_test = ImageNetWithOODEval(args.data_path, args.image_sub_path_test,
                                               transform=test_transforms)
        num_cls = dataset_train.num_classes
    elif args.dataset == 'part_imagenet':
        if args.eval_mode == 'val' or args.eval_mode == 'train':
            dataset_train = PartImageNetDataset(data_path=args.data_path, image_sub_path=args.image_sub_path_train,
                                                transform=train_transforms,
                                                annotation_file_path=args.anno_path_train)
            dataset_test = PartImageNetDataset(data_path=args.data_path, image_sub_path=args.image_sub_path_test,
                                               transform=test_transforms, annotation_file_path=args.anno_path_test,
                                               class_names=dataset_train.class_names,
                                               class_names_to_idx=dataset_train.class_names_to_idx,
                                               class_idx_to_names=dataset_train.class_idx_to_names)
        else:
            dataset_train_p1 = PartImageNetDataset(data_path=args.data_path,
                                                   image_sub_path=args.image_sub_path_train,
                                                   transform=train_transforms,
                                                   annotation_file_path=args.anno_path_train)
            p1_cls_num_list = dataset_train_p1.cls_num_list
            dataset_train_p2 = PartImageNetDataset(data_path=args.data_path, image_sub_path='val',
                                                   transform=train_transforms,
                                                   annotation_file_path=args.anno_path_train.replace('train', 'val'),
                                                   class_names=dataset_train_p1.class_names,
                                                   class_idx_to_names=dataset_train_p1.class_idx_to_names,
                                                   class_names_to_idx=dataset_train_p1.class_names_to_idx)
            p2_cls_num_list = dataset_train_p2.cls_num_list
            train_cls_num_list = [p1_cls_num_list[idx] + p2_cls_num_list[idx] for idx in
                                  range(len(p1_cls_num_list))]
            dataset_train = torch.utils.data.ConcatDataset([dataset_train_p1, dataset_train_p2])
            dataset_train.cls_num_list = train_cls_num_list
            dataset_test = PartImageNetDataset(data_path=args.data_path, image_sub_path=args.image_sub_path_test,
                                               transform=test_transforms, annotation_file_path=args.anno_path_test,
                                               class_names=dataset_train_p1.class_names,
                                               class_idx_to_names=dataset_train_p1.class_idx_to_names,
                                               class_names_to_idx=dataset_train_p1.class_names_to_idx)

        num_cls = dataset_test.num_classes

    elif args.dataset == 'part_imagenet_ood':

        dataset_train = PartImageNetDataset(data_path=args.data_path, image_sub_path=args.image_sub_path_train,
                                            transform=train_transforms,
                                            annotation_file_path=args.anno_path_train)
        dataset_test = PartImageNetDataset(data_path=args.data_path, image_sub_path=args.image_sub_path_test,
                                           transform=test_transforms, annotation_file_path=args.anno_path_test,
                                           class_names=dataset_train.class_names,
                                           class_names_to_idx=dataset_train.class_names_to_idx,
                                           class_idx_to_names=dataset_train.class_idx_to_names)
        num_cls = dataset_test.num_classes

    elif args.dataset == 'fgvc_aircraft':
        if args.eval_mode == 'val' or args.eval_mode == 'train':
            dataset_train = datasets.FGVCAircraft(root=args.data_path, split='train', transform=train_transforms,
                                                  target_transform=None, download=True)
            dataset_test = datasets.FGVCAircraft(root=args.data_path, split='val', transform=test_transforms,
                                                 target_transform=None, download=True)
        else:
            dataset_train = datasets.FGVCAircraft(root=args.data_path, split='trainval', transform=train_transforms,
                                                  target_transform=None, download=True)
            dataset_test = datasets.FGVCAircraft(root=args.data_path, split='test', transform=test_transforms,
                                                 target_transform=None, download=True)
        train_class_to_num_instances = Counter(dataset_train.targets)
        dataset_train.cls_num_list = [train_class_to_num_instances[idx] for idx in range(len(dataset_train.classes))]
        test_class_to_num_instances = Counter(dataset_test.targets)
        dataset_test.cls_num_list = [test_class_to_num_instances[idx] for idx in range(len(dataset_test.classes))]
        num_cls = len(dataset_test.classes)
    elif args.dataset == 'flowers102':
        if args.eval_mode == 'val' or args.eval_mode == 'train':
            dataset_train = datasets.Flowers102(root=args.data_path, split='train', transform=train_transforms,
                                                target_transform=None, download=True)
            dataset_test = datasets.Flowers102(root=args.data_path, split='val', transform=test_transforms,
                                               target_transform=None, download=True)
        else:
            dataset_train = datasets.Flowers102(root=args.data_path, split='train', transform=train_transforms,
                                                target_transform=None, download=True)
            dataset_test = datasets.Flowers102(root=args.data_path, split='test', transform=test_transforms,
                                               target_transform=None, download=True)
        num_cls = len(set(dataset_test._labels))
        train_class_to_num_instances = Counter(dataset_train._labels)
        dataset_train.cls_num_list = [train_class_to_num_instances[idx] for idx in range(num_cls)]
        test_class_to_num_instances = Counter(dataset_test._labels)
        dataset_test.cls_num_list = [test_class_to_num_instances[idx] for idx in range(num_cls)]
    elif args.dataset == 'plantnet':
        dataset_train = PlantNet(args.data_path, args.image_sub_path_train, transform=train_transforms,
                                 metadata_path=args.metadata_path,
                                 species_id_to_name_file=args.species_id_to_name_file)
        dataset_test = PlantNet(args.data_path, args.image_sub_path_test, transform=test_transforms,
                                metadata_path=args.metadata_path,
                                species_id_to_name_file=args.species_id_to_name_file)
        num_cls = dataset_test.num_classes
    else:
        raise ValueError('Dataset not supported.')
    return dataset_train, dataset_test, num_cls
