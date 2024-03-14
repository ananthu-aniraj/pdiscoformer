# Training Instructions

This document contains the instructions to train the models for the experiments in the paper.

The code has been designed to work with both single and multi-GPU training (including multi-node training) using PyTorch's Distributed Data Parallel (DDP) and the `torchrun` utility from [torchelastic](https://pytorch.org/docs/stable/elastic/run.html). It is also designed to auto-detect slurm environments and set the appropriate environment variables for multi-gpu training in DDP.

## Experiment Tracking
It is recommended to use [Weights and Biases](https://wandb.ai/site) for tracking the experiments. The `--wandb` flag can be used to enable this feature. Feel free to remove the `--wandb` flag if you don't want to use it.
The command line parameters in the training script related to Weights and Biases are as follows:
- `--wandb`: Enable Weights and Biases logging
- `--wandb_project`: Name of the project in Weights and Biases
- `--group`: Name of the experiment group within the project 
- `--job_type`: Name of the type of job within the experiment group
- `--wandb_entity`: Name of the entity in Weights and Biases. This is usually the username or the team name in Weights and Biases.
- `--wandb_mode`: Mode of logging in Weights and Biases. Use "offline" if the machine does not have internet access and "online" if it does. In case of "offline" mode, the logs can be uploaded to Weights and Biases later using the [wandb sync](https://docs.wandb.ai/ref/cli/wandb-sync) command.
- `--wandb_resume_id`: Resume a previous run in Weights and Biases. This is useful when you want to continue training from a previous run. Provide the run ID of the previous run to resume training.

## Batch Size and Learning Rate
The `--batch_size` and `--lr` parameters can be used to set the batch size and learning rate for training.
The batch size is per GPU, so the total batch size will be `batch_size * num_gpus`.

In case you want to modify the batch size, please adjust the learning rate according to the [square root scaling rule](https://arxiv.org/abs/1404.5997). 
We use a base batch size of 16 for a starting learning rate of 1e-6 (backbone).
So, if you want to use a batch size of 32, you should use a learning rate of 1e-6 * sqrt(32/16) = 1e-6 * sqrt(2) = 1.414e-6. 
The scaling is not implemented in the training script, so you will have to manually adjust the learning rate. 


## Training Commands
The training commands for the experiments in the paper are provided below. 






