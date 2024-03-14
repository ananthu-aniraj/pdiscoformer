# Training Instructions

This document contains the instructions to train the models for the experiments in the paper.

The code has been designed to work with both single and multi-GPU training (including multi-node training) using PyTorch`s Distributed Data Parallel (DDP) and the `torchrun` utility from [torchelastic](https://pytorch.org/docs/stable/elastic/run.html). It is also designed to auto-detect slurm environments and set the appropriate environment variables for multi-gpu training in DDP.

## Experiment Tracking
It is recommended to use [Weights and Biases](https://wandb.ai/site) for tracking the experiments. The `--wandb` flag can be used to enable this feature. Feel free to remove the `--wandb` flag if you don`t want to use it.
The command line parameters in the training script related to Weights and Biases are as follows:
- `--wandb`: Enable Weights and Biases logging
- `--wandb_project`: Name of the project in Weights and Biases
- `--group`: Name of the experiment group within the project 
- `--job_type`: Name of the type of job within the experiment group
- `--wandb_entity`: Name of the entity in Weights and Biases. This is usually the username or the team name in Weights and Biases.
- `--wandb_mode`: Mode of logging in Weights and Biases. Use "offline" if the machine does not have internet access and "online" if it does. In case of "offline" mode, the logs can be uploaded to Weights and Biases later using the [wandb sync](https://docs.wandb.ai/ref/cli/wandb-sync) command.
- `--wandb_resume_id`: Resume a previous run in Weights and Biases. This is useful when you want to continue training from a previous run. Provide the run ID of the previous run to resume training. Use this in combination with the `--resume_training` flag to resume training from a previous checkpoint.
- `--log_interval`: The interval at which the logs are printed plus the gradients are logged to Weights and Biases. The default value is 10. Feel free to change this value as required.

## Batch Size and Learning Rate
The `--batch_size` and `--lr` parameters can be used to set the batch size and learning rate for training.
The batch size is per GPU, so the total batch size will be `batch_size * num_gpus`.

In case you want to modify the batch size, please adjust the learning rate according to the [square root scaling rule](https://arxiv.org/abs/1404.5997). 
We use a base batch size of 16 for a starting learning rate of 1e-6 (backbone).
So, if you want to use a batch size of 32, you should use a learning rate of 1e-6 * sqrt(32/16) = 1e-6 * sqrt(2) = 1.414e-6. 
The scaling is not implemented in the training script, so you will have to manually adjust the learning rate. 


## Training Command
The main training command for the experiments in the paper are provided below. Please read the [Dataset-specific Parameters](#dataset-specific-parameters) and [Model-specific Parameters](#model-specific-parameters) sections to adjust the parameters as required for your experiments.

For example, to reproduce the training of the model on the CUB dataset for K=4 foreground parts, you can use the following command to train the model on a single node with 4 GPUs:
```
torchrun \
--nnodes=1 \
--nproc_per_node=4 \
<base path to the code>/train_net.py \
--model_arch vit_base_patch14_reg4_dinov2.lvd142m \
--pretrained_start_weights \
--data_path <base path to the dataset>/CUB_200_2011 \
--batch_size 8 \
--wandb \
--epochs 28 \
--dataset cub \
--save_every_n_epochs 16 \
--num_workers 2 \
--image_sub_path_train images \
--image_sub_path_test images \
--train_split 1 \
--eval_mode test \
--wandb_project <project name> \
--job_type <job name> \
--group <group name> \
--snapshot_dir <path to save directory> \
--lr 1.414e-6 \
--optimizer_type adam \
--scheduler_type steplr \
--scheduler_gamma 0.5 \
--scheduler_step_size 4 \
--scratch_lr_factor 1e4 \
--modulation_lr_factor 1e4 \
--finer_lr_factor 1e3 \
--drop_path 0.0 \
--smoothing 0 \
--augmentations_to_use cub_original \
--image_size 518 \
--num_parts 4 \
--weight_decay 0 \
--total_variation_loss 1.0 \
--concentration_loss 0.0 \
--enforced_presence_loss 2 \
--enforced_presence_loss_type enforced_presence \
--pixel_wise_entropy_loss 1.0 \
--gumbel_softmax \
--freeze_backbone \
--presence_loss_type original \
--modulation_type layer_norm \
--modulation_orth \
--grad_norm_clip 2.0
```

### Dataset-specific Parameters
- `--dataset`: The name of the dataset. For CUB, use `cub`. For PartImageNet OOD, use `part_imagenet_ood`. For Oxford Flowers, use `flowers102`. For PartImageNet Seg, use `part_imagenet`. Even more datasets are supported, for more details, please refer to the [load_dataset](load_dataset.py) file.
- `--data_path`: The path to the dataset. The folder structure should be as mentioned in the [README](README.md) file.
- `--image_sub_path_train`: The sub-path to the training images in the dataset. For instance, in the CUB dataset, the images are present in the `images` folder.
- `--image_sub_path_test`: The sub-path to the test images in the dataset. 
- `--train_split`: The split to use for training. Only applicable for CUB if you wish to train on a subset of the dataset. We always use the full dataset for training in the paper, so the default value is 1.
- `--eval_mode`: The mode for evaluation. Use `test` for evaluation on the test set and `val` for evaluation on the validation set. The default value is `test`. All the experiments in the paper are evaluated on the test set.
- `--augmentations_to_use`: The augmentations to use for training. The augmentations are defined in the [transform_utils](utils/data_utils/transform_utils.py) file. The default value is `cub_original` which uses standard augmentations from fine-grained classification literature. We also support a more sophisticated auto-augmentation policy, which can be used by setting the value to `timm`. This uses the auto-augment policy used by the [ConvNeXt paper](https://arxiv.org/abs/2201.03545). For all the experiments in our paper, we use the `cub_original` augmentations.
- `--image_size`: The size of the input image. This value is set to 518 (default value for DinoV2 timm models) on CUB for the ViT models. For the ResNet models on CUB, the default value is 448 (same as in related works). For the other datasets, the image size is set to a constant value of 224 for both ViT and ResNet models.
- `--anno_path_train`: The path to the training annotation file for the PartImageNet OOD/ PartImageNet Seg datasets. In case of PartImageNet OOD, the annotation file is generated using the pre-processing script mentioned in the [README](README.md) file. Not applicable for other datasets.
- `--anno_path_test`: The path to the test annotation file for the PartImageNet OOD/ PartImageNet Seg datasets. In case of PartImageNet OOD, this is also generated using the pre-processing script mentioned in the [README](README.md) file. Not applicable for other datasets.
- `--metadata_path`: The path to the metadata file for the [PlantNet300K](https://zenodo.org/records/5645731) dataset. Not applicable for other datasets. Due to an absence of part annotations, we do not use the dataset in the paper. However, the code supports the dataset for future use.
- `--species_id_to_name_file`: The path to the file containing the mapping of species IDs to species names for the [PlantNet300K](https://zenodo.org/records/5645731) dataset. Not applicable for other datasets. 
- `--turn_on_mixup_or_cutmix`: Turn on mixup or cutmix for training. We do not use this in the paper for any of the experiments, so the default value is `False`.


### Model-specific Parameters
- `--model_arch`: The architecture of the model. For the experiments in the paper, we use the [ViT-Base DinoV2](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) model. In theory, any model from the timm library supported by the [VisionTransformer class](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py) can be used. Additionally, we also support all the torchvision and timm ResNet models and the timm ConvNeXt models. 
- `--num_parts`: The number of foreground parts to predict. Please adjust this value as required for the dataset. It is recommended to use the values specified in the paper. 
- `--pretrained_start_weights`: Use the pre-trained weights for the backbone. This requires an active internet connection. If you want to train from scratch, you can remove this flag.
- `--use_torchvision_resnet_model`: Use the torchvision implementation of the ResNet model. This is used for the ResNet models. If you want to use the timm implementation, you can remove this flag.
- `--freeze_backbone`: Freeze the backbone weights. This will freeze all the layers in the ViT backbone, except for the layers we introduce for part discovery, the class, register tokens and position embeddings. This is used for the experiments in the paper. For ResNet and ConvNeXt models, this flag will freeze the entire backbone except for the part discovery layers.
- `--freeze_params`: Only applicable for the ViT models. In combination with the `--freeze_backbone` flag, this can be used to freeze the entire model except for the part discovery layers. This can be used to reproduce our results for the fully frozen ViT model.
- `--modulation_type`: The type of modulation to use for the part discovery layers. The default value is `layer_norm`. We use this value for all the experiments in the paper.
- `--modulation_orth`: Apply orthogonality loss on the modulated features. This is used for the experiments in the paper.
- `--gumbel_softmax`: Use the Gumbel-Softmax trick on the part attention maps. This is used for the experiments in the paper.
- `--gumbel_softmax_temperature`: The temperature for the Gumbel-Softmax trick. The default value is 1.0. We use this value for all the experiments in the paper.
- `--gumbel_softmax_hard`: Use the straight-through estimator for the Gumbel-Softmax. This is not used for the experiments in the paper.
- `--classifier_type`: The type of classifier to use for the class predictions. We use the default value `linear` for all the experiments in the paper.
- `--part_dropout`: The dropout probability for the part dropout. The default value is 0.3. We use this value for all the experiments in the paper.
- `--drop_path`: The drop path probability for the ViT models. The default value is 0.0. We use this value for all the experiments in the paper. This is not used unless you fully fine-tune the model.
- `--noise_variance`: The variance of the Gaussian noise to add to the part attention maps. We do not use this in the paper, so the default value is 0.0.
- `--grad_norm_clip`: The maximum norm for the gradients. The default value is 2.0. We use this value for all the experiments in the paper.
- `--output_stride`: Only applicable if you use CNN models from timm. 

### Checkpointing and Logging Parameters 
- `--snapshot_dir`: The directory to save the checkpoints. Feel free to change this value as required.
- `--save_every_n_epochs`: The interval at which the checkpoints as well as (optionally) part assignment maps are saved. The default value is 16. Feel free to change this value as required. By default, the checkpoint with the best validation accuracy and the last checkpoint are saved. We use the model with the best validation accuracy for evaluation in the paper. 
- `--amap_saving_prob`: The probability of saving the part assignment maps. This is triggered on the first epoch, every save_every_n_epochs epoch and the last epoch. Set it to 0 to turn it off and 1 if you want to save it for every iteration. We recommend using a value of 0.05 during training and higher values such as 0.8 for evaluation. This can cause a significant slowdown during training as the maps are saved as images. 

## Optimizer and Scheduler Parameters
- `--optimizer_type`: The type of optimizer to use. The default value is `adam`. We use this value for all the experiments in the paper.
- `--scheduler_type`: The type of scheduler to use. The default value is `steplr`. We use this value for all the experiments in the paper.
- `--scheduler_gamma`: The gamma value for the scheduler. The default value is 0.5. We use this value for all the experiments in the paper.
- `--scheduler_step_size`: The step size for the scheduler. The default value is 4. We use this value for all the experiments in the paper.
- `--lr`: The learning rate for training. Please refer to the [Batch Size and Learning Rate](#batch-size-and-learning-rate) section for more details.
- `--weight_decay`: The weight decay for the optimizer. The default value is 0. We use this value for all the experiments in the paper. We have implemented the normalized weight decay formulation from the [AdamW paper](https://arxiv.org/abs/1711.05101) in the code.
- `--scratch_lr_factor`: The learning rate factor for the scratch layers. The default value is 1e4. We use this value for all the experiments in the paper.
- `--modulation_lr_factor`: The learning rate factor for the modulation layers. The default value is 1e4. We use this value for all the experiments in the paper.
- `--finer_lr_factor`: The learning rate factor for the finer layers. The default value is 1e3. We use this value for all the experiments in the paper.

### Loss Hyper-Parameters
The loss hyperparameters are already set to the values used in the paper. See the [training arguments script](argument_parser_train.py) for more details.
Feel free to modify these values if it is required for your experiments.

### Extra Notes
- If you wish to train on a single GPU, you can remove the `torchrun` command and the `--nnodes` and `--nproc_per_node` flags. Then run it as `python <base path to the code>/train_net.py <arguments>`.
- Please note that the code is written with the assumption that all the visible GPUs are to be used for training. If you want to use a subset of the GPUs, you will have to manually set the `CUDA_VISIBLE_DEVICES` environment variable before running the training script. This is automatically done in slurm and other job schedulers environments, so you don`t have to worry about it in those cases.
- The `--pretrained_start_weights` flag is used to load the pre-trained weights for the backbone. This requires an internet connection to download the weights from the timm library. If you want to train from scratch, you can remove this flag.
-  The weights will be saved in the `~/.cache/torch/hub/checkpoints` directory which is automatically detected in our code, if already present.
- For ResNet models, we support both the timm and torchvision implementations. However, please note that all the works on part discovery in the literature, to the best of our knowledge, use the torchvision weights.
 If you wish to use the torchvision implementation, you can use the `--use_torchvision_resnet_model` flag.  
- (OPTIONAL) If you do not have an active internet connection, it is also possible to separately run the `create_model()` function [here](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) to download the weights for the vit-base model. This will auto-detect the `~/.cache/torch/hub/checkpoints` directory and save the weights there. Similarly, for the resnet model, use the [get_model function](https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models) for torchvision with `weights="DEFAULT"` or the `create_model()` [here](https://huggingface.co/timm/resnet101.a1h_in1k) for the timm weights.



