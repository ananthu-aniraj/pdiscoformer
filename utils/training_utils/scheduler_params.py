from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from .linear_lr_scheduler import LinearLRScheduler


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
    decay_steps = args.scheduler_step_size
    warmup_lr_init = args.warmup_lr

    restart_factor = args.scheduler_restart_factor
    gamma = args.scheduler_gamma

    min_lr = args.min_lr
    if type_lr_schedule == 'cosine':
        return CosineLRScheduler(
            optimizer,
            t_initial=total_steps,
            cycle_decay=restart_factor,
            lr_min=min_lr,
            warmup_t=warmup_steps,
            cycle_limit=args.cosine_cycle_limit,
            warmup_lr_init=warmup_lr_init,
            t_in_epochs=True
        )
    elif type_lr_schedule == 'steplr':
        return StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=gamma,
            warmup_t=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            t_in_epochs=True
        )
    elif type_lr_schedule == 'linearlr':
        return LinearLRScheduler(
            optimizer,
            t_initial=total_steps,
            lr_min_rate=0.01,
            warmup_t=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            t_in_epochs=True
        )
    else:
        raise NotImplementedError

