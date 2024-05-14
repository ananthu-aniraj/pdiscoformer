import torch
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from pytopk import ImbalNoisedTopK


def load_classification_loss(args, dataset_train, num_cls):
    """
    Load the loss function for classification
    :param args: Arguments from the argument parser
    :param dataset_train: Training dataset
    :param num_cls: Number of classes in the dataset
    :return:
    loss_fn: List of loss functions for training and evaluation
    """
    # Mixup/Cutmix
    mixup_fn = None
    mixup_active = args.turn_on_mixup_or_cutmix
    if mixup_active:
        print("Mixup is activated! Please note that this may not work with the equivariance loss")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_cls)

    if args.use_imbalanced_noised_topk:
        loss_fn_train = ImbalNoisedTopK(k=args.topk_k, epsilon=args.topk_epsilon, max_m=args.max_m,
                                        cls_num_list=dataset_train.cls_num_list, scale=args.topk_scale,
                                        n_sample=args.topk_n_sample)
        print(
            "Using Imbalanced Noised TopK loss, please note that label smoothing and mixup are not implemented in this case.")
        mixup_fn = None
    else:
        # Define loss and optimizer
        if mixup_fn is not None:
            # smoothing is handled with mix-up label transform
            loss_fn_train = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            loss_fn_train = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            loss_fn_train = torch.nn.CrossEntropyLoss()

    loss_fn_eval = torch.nn.CrossEntropyLoss()
    loss_fn = [loss_fn_train, loss_fn_eval]
    return loss_fn, mixup_fn


def load_loss_hyper_params(args):
    """
    Load the hyperparameters for the loss functions and affine transform parameters for equivariance
    :param args: Arguments from the argument parser
    :return:
    loss_hyperparams: Dictionary of loss hyperparameters
    eq_affine_transform_params: Dictionary of affine transform parameters for equivariance
    """
    loss_hyperparams = {'l_class_att': args.classification_loss, 'l_presence': args.presence_loss,
                        'l_presence_beta': args.presence_loss_beta, 'l_presence_type': args.presence_loss_type,
                        'l_equiv': args.equivariance_loss, 'l_conc': args.concentration_loss,
                        'l_orth': args.orthogonality_loss_landmarks, 'l_tv': args.total_variation_loss,
                        'l_enforced_presence': args.enforced_presence_loss, 'l_pixel_wise_entropy': args.pixel_wise_entropy_loss,
                        'l_enforced_presence_loss_type': args.enforced_presence_loss_type}

    # Affine transform parameters for equivariance
    degrees = [-args.degrees, args.degrees]
    translate = [args.translate_x, args.translate_y]
    scale = [args.scale_l, args.scale_u]
    shear_x = args.shear_x
    shear_y = args.shear_y
    shear = [shear_x, shear_y]
    if shear_x == 0.0 and shear_y == 0.0:
        shear = None

    eq_affine_transform_params = {'degrees': degrees, 'translate': translate, 'scale_ranges': scale, 'shear': shear}

    return loss_hyperparams, eq_affine_transform_params
