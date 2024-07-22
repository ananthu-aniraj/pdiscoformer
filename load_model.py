import copy
import os
from pathlib import Path

import torch
from timm.models import create_model
from torchvision.models import get_model

from models import pdiscoformer_vit_bb, pdisconet_vit_bb, pdisconet_resnet_torchvision_bb
from models.individual_landmark_resnet import IndividualLandmarkResNet
from models.individual_landmark_convnext import IndividualLandmarkConvNext
from models.individual_landmark_vit import IndividualLandmarkViT
from utils import load_state_dict_pdisco


def load_model_arch(args, num_cls):
    """
    Function to load the model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    if 'resnet' in args.model_arch:
        num_layers_split = [int(s) for s in args.model_arch if s.isdigit()]
        num_layers = int(''.join(map(str, num_layers_split)))
        if num_layers >= 100:
            timm_model_arch = args.model_arch + ".a1h_in1k"
        else:
            timm_model_arch = args.model_arch + ".a1_in1k"

    if "resnet" in args.model_arch and args.use_torchvision_resnet_model:
        weights = "DEFAULT" if args.pretrained_start_weights else None
        base_model = get_model(args.model_arch, weights=weights)
    elif "resnet" in args.model_arch and not args.use_torchvision_resnet_model:
        if args.eval_only:
            base_model = create_model(
                timm_model_arch,
                pretrained=args.pretrained_start_weights,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
        else:
            base_model = create_model(
                timm_model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )

    elif "convnext" in args.model_arch:
        if args.eval_only:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
        else:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
    elif "vit" in args.model_arch:
        if args.eval_only:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                img_size=args.image_size,
            )
        else:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                img_size=args.image_size,
            )
        vit_patch_size = base_model.patch_embed.proj.kernel_size[0]
        if args.image_size % vit_patch_size != 0:
            raise ValueError(f"Image size {args.image_size} must be divisible by patch size {vit_patch_size}")
    else:
        raise ValueError('Model not supported.')

    return base_model


def init_pdisco_model(base_model, args, num_cls):
    """
    Function to initialize the model
    :param base_model: Base model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    # Initialize the network
    if 'convnext' in args.model_arch:
        sl_channels = base_model.stages[-1].downsample[-1].in_channels
        fl_channels = base_model.head.in_features
        model = IndividualLandmarkConvNext(base_model, args.num_parts, num_classes=num_cls,
                                           sl_channels=sl_channels, fl_channels=fl_channels,
                                           part_dropout=args.part_dropout, modulation_type=args.modulation_type,
                                           gumbel_softmax=args.gumbel_softmax,
                                           gumbel_softmax_temperature=args.gumbel_softmax_temperature,
                                           gumbel_softmax_hard=args.gumbel_softmax_hard,
                                           modulation_orth=args.modulation_orth, classifier_type=args.classifier_type,
                                           noise_variance=args.noise_variance)
    elif 'resnet' in args.model_arch:
        sl_channels = base_model.layer4[0].conv1.in_channels
        fl_channels = base_model.fc.in_features
        model = IndividualLandmarkResNet(base_model, args.num_parts, num_classes=num_cls,
                                         sl_channels=sl_channels, fl_channels=fl_channels,
                                         use_torchvision_model=args.use_torchvision_resnet_model,
                                         part_dropout=args.part_dropout, modulation_type=args.modulation_type,
                                         gumbel_softmax=args.gumbel_softmax,
                                         gumbel_softmax_temperature=args.gumbel_softmax_temperature,
                                         gumbel_softmax_hard=args.gumbel_softmax_hard,
                                         modulation_orth=args.modulation_orth, classifier_type=args.classifier_type,
                                         noise_variance=args.noise_variance)
    elif 'vit' in args.model_arch:
        model = IndividualLandmarkViT(base_model, num_landmarks=args.num_parts, num_classes=num_cls,
                                      part_dropout=args.part_dropout,
                                      modulation_type=args.modulation_type, gumbel_softmax=args.gumbel_softmax,
                                      gumbel_softmax_temperature=args.gumbel_softmax_temperature,
                                      gumbel_softmax_hard=args.gumbel_softmax_hard,
                                      modulation_orth=args.modulation_orth, classifier_type=args.classifier_type,
                                      noise_variance=args.noise_variance)
    else:
        raise ValueError('Model not supported.')

    return model


def load_model_pdisco(args, num_cls):
    """
    Function to load the model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    base_model = load_model_arch(args, num_cls)
    model = init_pdisco_model(base_model, args, num_cls)

    return model


def pdiscoformer_vit(pretrained=True, model_dataset="cub", k=8, model_url="", img_size=224, num_cls=200):
    """
    Function to load the PDiscoFormer model with ViT backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :param model_url: URL to load the model weights from
    :param img_size: Image size
    :param num_cls: Number of classes in the dataset
    :return: PDiscoFormer model with ViT backbone
    """
    model = pdiscoformer_vit_bb("vit_base_patch14_reg4_dinov2.lvd142m", num_cls=num_cls, k=k, img_size=img_size)
    if pretrained:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "pdiscoformer_checkpoints", f"pdiscoformer_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_pdisco(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model


def pdisconet_vit(pretrained=True, model_dataset="nabirds", k=8, model_url="", img_size=224, num_cls=555):
    """
    Function to load the PDiscoNet model with ViT backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :param model_url: URL to load the model weights from
    :param img_size: Image size
    :param num_cls: Number of classes in the dataset
    :return: PDiscoNet model with ViT backbone
    """
    model = pdisconet_vit_bb("vit_base_patch14_reg4_dinov2.lvd142m", num_cls=num_cls, k=k, img_size=img_size)
    if pretrained:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "pdiscoformer_checkpoints", f"pdisconet_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_pdisco(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model


def pdisconet_resnet101(pretrained=True, model_dataset="nabirds", k=8, model_url="", num_cls=555):
    """
    Function to load the PDiscoNet model with ResNet-101 backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :param model_url: URL to load the model weights from
    :param num_cls: Number of classes in the dataset
    :return: PDiscoNet model with ResNet-101 backbone
    """
    model = pdisconet_resnet_torchvision_bb("resnet101", num_cls=num_cls, k=k)
    if pretrained:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "pdiscoformer_checkpoints", f"pdisconet_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_pdisco(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model
