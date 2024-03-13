import torch
from timm.models import create_model
from torchvision.models import get_model
from models.individual_landmark_resnet import IndividualLandmarkResNet
from models.individual_landmark_convnext import IndividualLandmarkConvNext
from models.individual_landmark_vit import IndividualLandmarkViT


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
