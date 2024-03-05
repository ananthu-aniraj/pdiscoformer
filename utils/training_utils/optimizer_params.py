import torch
import math
from timm.optim.lars import Lars
from timm.optim.lamb import Lamb
from utils.training_utils.ddp_utils import calculate_effective_batch_size


def build_optimizer(args, params_groups, dataset_train):
    """
    Function to build the optimizer
    :param args: arguments from the command line
    :param params_groups: parameters to be optimized
    :param dataset_train: training dataset
    :return: optimizer
    """
    grad_averaging = not args.turn_off_grad_averaging
    weight_decay = calculate_weight_decay(args, dataset_train)
    print(f'Weight decay in current training: {weight_decay:.6f}')
    if args.optimizer_type == 'adamw':
        return torch.optim.AdamW(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                 weight_decay=weight_decay)
    elif args.optimizer_type == 'sgd':
        return torch.optim.SGD(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                               nesterov=True)
    elif args.optimizer_type == 'adam':
        return torch.optim.Adam(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                weight_decay=weight_decay)
    elif args.optimizer_type == 'nadam':
        return torch.optim.NAdam(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                 weight_decay=weight_decay)
    elif args.optimizer_type == 'lars':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, trust_coeff=args.trust_coeff, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'nlars':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, nesterov=True, trust_coeff=args.trust_coeff, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'larc':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, trust_coeff=args.trust_coeff, trust_clip=True,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'nlarc':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, nesterov=True, trust_coeff=args.trust_coeff, trust_clip=True,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'lamb':
        return Lamb(params=params_groups, lr=args.lr, betas=(args.betas1, args.betas2), weight_decay=weight_decay,
                    grad_averaging=grad_averaging, max_grad_norm=args.max_grad_norm, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'lambc':
        return Lamb(params=params_groups, lr=args.lr, betas=(args.betas1, args.betas2), weight_decay=weight_decay,
                    grad_averaging=grad_averaging, max_grad_norm=args.max_grad_norm, trust_clip=True,
                    always_adapt=args.always_adapt)
    else:
        raise NotImplementedError(f'Optimizer {args.optimizer_type} not implemented.')


def calculate_weight_decay(args, dataset_train):
    """
    Function to calculate the weight decay
    :param args: Arguments from the command line
    :param dataset_train: Training dataset
    :return: weight_decay: Weight decay
    """
    batch_size = calculate_effective_batch_size(args)
    num_iterations = len(dataset_train) // batch_size  # Since we set drop_last=True
    norm_weight_decay = args.weight_decay
    weight_decay = norm_weight_decay * math.sqrt(1 / (num_iterations * args.epochs))
    return weight_decay


def layer_group_matcher_pdisconet(args, model):
    """
    Function to group the parameters of the model into different groups
    :param args: Arguments from the command line
    :param model: Model to be trained
    :return: param_groups: Parameters grouped into different groups
    """
    scratch_layers = ["fc_class_landmarks"]
    modulation_layers = ["modulation", "modulation_parts", "modulation_instances"]
    finer_layers = ["fc_landmarks", "fc_landmarks_instances", "decoder", "landmark_tokens"]
    unfrozen_layers = ["cls_token", "pos_embed", "reg_token"]
    scratch_parameters = []
    modulation_parameters = []
    backbone_parameters_wd = []
    no_weight_decay_params = []
    finer_parameters = []

    for name, p in model.named_parameters():
        if any(x in name for x in scratch_layers):
            print("scratch layer_name: " + name)
            scratch_parameters.append(p)
            p.requires_grad = True

        elif any(x in name for x in modulation_layers):
            print("modulation layer_name: " + name)
            modulation_parameters.append(p)
            p.requires_grad = True

        elif any(x in name for x in finer_layers):
            print("finer layer_name: " + name)
            finer_parameters.append(p)
            p.requires_grad = True

        elif any(x in name for x in unfrozen_layers):
            no_weight_decay_params.append(p)
            if args.freeze_params:
                p.requires_grad = False
            else:
                print("unfrozen layer_name: " + name)
                p.requires_grad = True

        else:
            if args.freeze_backbone:
                p.requires_grad = False
            else:
                p.requires_grad = True

            if p.ndim == 1:
                no_weight_decay_params.append(p)
            else:
                backbone_parameters_wd.append(p)

    param_groups = [{'params': backbone_parameters_wd, 'lr': args.lr},
                    {'params': no_weight_decay_params, 'lr': args.lr, 'weight_decay': 0.0},
                    {'params': finer_parameters, 'lr': args.lr * args.finer_lr_factor, 'weight_decay': 0.0},
                    {'params': modulation_parameters, 'lr': args.lr * args.modulation_lr_factor, 'weight_decay': 0.0},
                    {'params': scratch_parameters, 'lr': args.lr * args.scratch_lr_factor}]

    return param_groups


def layer_group_matcher_baseline(args, model):
    """
    Function to group the parameters of the model into different groups
    :param args: Arguments from the command line
    :param model: Model to be trained
    :return: param_groups: Parameters grouped into different groups
    """
    scratch_layers = ["head", "fc"]
    scratch_parameters = []
    no_weight_decay_params_scratch = []
    finetune_parameters = []
    no_weight_decay_params_bb = []
    for name, p in model.named_parameters():

        if any(x in name for x in scratch_layers):
            print("scratch layer_name: " + name)
            if p.ndim == 1:
                no_weight_decay_params_scratch.append(p)
            else:
                scratch_parameters.append(p)
        else:
            if p.ndim == 1:
                no_weight_decay_params_bb.append(p)
            else:
                finetune_parameters.append(p)
            if args.freeze_backbone:
                p.requires_grad = False
            else:
                p.requires_grad = True

    param_groups = [{'params': finetune_parameters, 'lr': args.lr},
                    {'params': no_weight_decay_params_bb, 'lr': args.lr, 'weight_decay': 0.0},
                    {'params': scratch_parameters, 'lr': args.lr * args.scratch_lr_factor}]
    if len(no_weight_decay_params_scratch) > 0:
        param_groups.append(
            {'params': no_weight_decay_params_scratch, 'lr': args.lr * args.scratch_lr_factor, 'weight_decay': 0.0})
    return param_groups
