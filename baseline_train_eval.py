import torch
from timeit import default_timer as timer

from argument_parser_baseline import parse_args
from load_dataset import get_dataset
from load_model import load_model_baseline
from load_losses import load_classification_loss
from utils.data_utils.transform_utils import load_transforms
from utils.misc_utils import sync_bn_conversion, check_snapshot
from utils.wandb_params import get_train_loggers
from utils.training_utils.optimizer_params import build_optimizer, layer_group_matcher_baseline
from utils.training_utils.scheduler_params import build_scheduler
from utils.training_utils.ddp_utils import multi_gpu_check
from engine.training_baseline import baseline_trainer

torch.backends.cudnn.benchmark = True


def baseline_train_eval():
    args = parse_args()

    train_loggers = get_train_loggers(args)

    # Create directory to save training checkpoints, otherwise load the existing checkpoint
    check_snapshot(args)

    # Get the transforms and load the dataset
    train_transforms, test_transforms = load_transforms(args)
    dataset_train, dataset_test, num_cls = get_dataset(args, train_transforms, test_transforms)

    # Load the model
    model = load_model_baseline(args, num_cls)

    # Check if there are multiple GPUs
    use_ddp = multi_gpu_check()
    # Convert BatchNorm to SyncBatchNorm if there is more than 1 GPU
    if use_ddp:
        model = sync_bn_conversion(model)

    # Load the loss function
    loss_fn, mixup_fn = load_classification_loss(args, dataset_train, num_cls)

    # Load the optimizer and scheduler
    param_groups = layer_group_matcher_baseline(args, model)
    optimizer = build_optimizer(args, param_groups, dataset_train)
    scheduler = build_scheduler(args, optimizer)

    # Start the timer
    start_time = timer()

    # Setup training and save the results
    baseline_trainer(model=model,
                     train_dataset=dataset_train,
                     test_dataset=dataset_test,
                     batch_size=args.batch_size,
                     optimizer=optimizer,
                     scheduler=scheduler,
                     loss_fn=loss_fn,
                     epochs=args.epochs,
                     save_every=args.save_every_n_epochs,
                     loggers=train_loggers,
                     log_freq=args.log_interval,
                     use_amp=args.use_amp,
                     snapshot_path=args.snapshot_dir,
                     grad_norm_clip=args.grad_norm_clip,
                     num_workers=args.num_workers,
                     mixup_fn=mixup_fn,
                     seed=args.seed,
                     eval_only=args.eval_only,
                     use_ddp=use_ddp,
                     class_balanced_sampling=args.use_class_balanced_sampling,
                     num_samples_per_class=args.num_samples_per_class,
                     )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    baseline_train_eval()
