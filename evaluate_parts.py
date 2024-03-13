"""
From: https://github.com/zxhuang1698/interpretability-by-parts/blob/master/src/cub200/eval_interp.py
"""
# pytorch & misc
import torch
import torchvision.transforms as transforms
from data_sets import CUB200, PartImageNetDataset, Flowers102Seg
from load_model import load_model_pdisco
import argparse
import copy
import os
from engine.eval_interpretability_nmi_ari_keypoint import eval_nmi_ari, eval_kpr
from engine.eval_fg_bg import FgBgIoU
from utils.training_utils.engine_utils import load_state_dict_pdisco

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate model interpretability via part parsing quality'
    )
    parser.add_argument('--model_arch', default='resnet50', type=str,
                        help='pick model architecture')
    parser.add_argument('--use_torchvision_resnet_model', default=False, action='store_true')

    # Data
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"', required=True)
    parser.add_argument('--image_sub_path', default='images', type=str, required=False)
    parser.add_argument('--dataset', default='cub', type=str)
    parser.add_argument('--anno_path_test', default='', type=str, required=False)
    parser.add_argument('--center_crop', default=False, action='store_true')

    # Eval mode
    parser.add_argument('--eval_mode', default='keypoint', choices=['keypoint', 'nmi_ari', 'fg_bg'], type=str)

    # Model params
    parser.add_argument('--num_parts', help='number of parts to predict',
                        default=8, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--output_stride', default=32, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    # Modulation
    parser.add_argument('--modulation_type', default="original",
                        choices=["original", "layer_norm", "parallel_mlp", "parallel_mlp_no_bias",
                                 "parallel_mlp_no_act", "parallel_mlp_no_act_no_bias", "none"],
                        type=str)
    parser.add_argument('--modulation_orth', default=False, action='store_true',
                        help='use orthogonality loss on modulated features')
    # Part Dropout
    parser.add_argument('--part_dropout', default=0.0, type=float)

    # Add noise to vit output features
    parser.add_argument('--noise_variance', default=0.0, type=float)

    # Gumbel Softmax
    parser.add_argument('--gumbel_softmax', default=False, action='store_true')
    parser.add_argument('--gumbel_softmax_temperature', default=1.0, type=float)
    parser.add_argument('--gumbel_softmax_hard', default=False, action='store_true')

    # Model path
    parser.add_argument('--model_path', default=None, type=str)

    # Classifier type
    parser.add_argument('--classifier_type', default="linear",
                        choices=["linear", "independent_mlp"], type=str)

    args = parser.parse_args()
    return args


def main(args):
    mode = args.eval_mode
    nparts = args.num_parts
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resize_transform = transforms.Resize(size=args.image_size)
    resize_transform_mask = transforms.Resize(size=args.image_size, interpolation=transforms.InterpolationMode.NEAREST)
    center_crop_transform = transforms.CenterCrop(size=args.image_size)
    def_transform = transforms.ToTensor()
    if "vit" in args.model_arch:
        if not args.center_crop:
            raise ValueError('ViT models require center crop.')

    if args.center_crop and args.dataset != 'cub':
        data_transforms = transforms.Compose([resize_transform, center_crop_transform, def_transform])
        mask_transform = transforms.Compose([resize_transform_mask, center_crop_transform])

    else:
        data_transforms = transforms.Compose([resize_transform, def_transform])
        mask_transform = resize_transform_mask

    # define dataset path
    if args.dataset == 'cub':
        cub_path = args.data_path
        # define dataset and loader
        eval_data = CUB200(cub_path,
                           train=False, transform=data_transforms, resize=args.image_size, center_crop=args.center_crop,
                           image_sub_path=args.image_sub_path)
    elif args.dataset == 'part_imagenet':
        # define dataset and loader
        eval_data = PartImageNetDataset(data_path=args.data_path, image_sub_path=args.image_sub_path,
                                        transform=data_transforms,
                                        annotation_file_path=args.anno_path_test,
                                        get_masks=True,
                                        mask_transform=mask_transform,
                                        )

    elif args.dataset == 'flowers102seg':
        # define dataset and loader
        eval_data = Flowers102Seg(args.data_path, transform=data_transforms, mask_transform=mask_transform, split='test')

    else:
        raise ValueError('Dataset not supported.')

    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    num_cls = eval_data.num_classes
    if args.dataset != 'flowers102seg':
        num_landmarks = eval_data.num_kps

    # Add arguments to args
    args.eval_only = True
    args.pretrained_start_weights = True
    # Load the model
    net = load_model_pdisco(args, num_cls)
    snapshot_data = torch.load(args.model_path, map_location=torch.device('cpu'))
    if 'model_state' in snapshot_data:
        _, state_dict = load_state_dict_pdisco(snapshot_data)
    else:
        state_dict = copy.deepcopy(snapshot_data)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    net.to(device)

    if mode == 'keypoint':
        if args.dataset == 'cub':
            fit_data = CUB200(cub_path,
                              train=True, transform=data_transforms, resize=args.image_size,
                              center_crop=args.center_crop)
            fit_loader = torch.utils.data.DataLoader(
                fit_data, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)
            kpr = eval_kpr(net, fit_loader, eval_loader, nparts, num_landmarks=num_landmarks)
            print('Mean keypoint regression error on the test set is %.2f%%.' % kpr)
        else:
            raise ValueError('Dataset not supported.')

    elif mode == 'nmi_ari':
        nmi, ari = eval_nmi_ari(net, eval_loader, dataset=args.dataset)
        print(nmi)
        print(ari)
        print('NMI between predicted and ground truth parts is %.2f' % nmi)
        print('ARI between predicted and ground truth parts is %.2f' % ari)
        print('Evaluation finished.')

    elif mode == 'fg_bg':
        if args.dataset != 'flowers102seg':
            raise ValueError('Dataset not supported.')
        iou_calculator = FgBgIoU(net, eval_loader, device)
        iou_calculator.calculate_iou(args.model_path)
        m_iou = iou_calculator.metric_fg.compute().item() * 100
        m_iou_bg = iou_calculator.metric_bg.compute().item() * 100
        print('Foreground mIoU is %.2f' % m_iou)
        print('Background mIoU is %.2f' % m_iou_bg)
        print('Evaluation finished.')

    else:
        print("Please run with either keypoint or nmi_ari or fg_bg mode.")


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
