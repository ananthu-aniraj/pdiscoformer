import torch

from utils.training_utils.cosine_scheduler_w_linear_warmup import LinearWarmupCosineAnnealingLR


def build_scheduler(args, optimizer):
    """
    Function to build the scheduler
    :param args: arguments from the command line
    :param optimizer: optimizer used for training
    :return: scheduler
    """
    # initialize scheduler hyperparameters
    total_steps = args.epochs
    type_lr_schedule = args.scheduler_type
    warmup_steps = args.scheduler_warmup_epochs
    start_factor = args.scheduler_start_factor
    end_factor = args.scheduler_end_factor
    restart_factor = args.scheduler_restart_factor
    gamma = args.scheduler_gamma
    step_size = args.scheduler_step_size
    min_lr = args.min_lr

    if type_lr_schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
    elif type_lr_schedule == 'cosine_warmup_restart':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=restart_factor,
                                                                    eta_min=min_lr)
    elif type_lr_schedule == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif type_lr_schedule == 'linearlr':
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor,
                                                 total_iters=total_steps)
    elif type_lr_schedule == 'cosine_with_warmup':
        return LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_steps, max_epochs=total_steps,
                                             eta_min=min_lr, warmup_start_lr=args.warmup_lr)
    else:
        raise NotImplementedError
