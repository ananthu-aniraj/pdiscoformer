import wandb
import copy


def init_wandb(args):
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)
    if args["resume_training"]:
        if args_dict["wandb_resume_id"] is not None:
            run = wandb.init(project=args_dict["wandb_project"], entity=args_dict["wandb_entity"],
                             job_type=args_dict["job_type"],
                             group=args_dict["group"], mode=args_dict["wandb_mode"],
                             config=args_dict, id=args_dict["wandb_resume_id"], resume="must")
        else:
            raise ValueError("wandb_resume_id is None")
    else:
        run = wandb.init(project=args_dict["wandb_project"], entity=args_dict["wandb_entity"],
                         job_type=args_dict["job_type"],
                         group=args_dict["group"], mode=args_dict["wandb_mode"],
                         config=args_dict)
    return run


def get_train_loggers(args):
    """Get the train loggers for the experiment"""
    train_loggers = []
    if args.wandb:
        wandb_logger_settings = copy.deepcopy(vars(args))
        train_loggers.append(wandb_logger_settings)
    return train_loggers
