# Evaluation Instructions
- We recommend evaluating on one GPU. The code technically runs for multiple GPUs as well, but we have not implemented the final averaging of the evaluation metrics across GPUs.
- Additionally, we observe that it is best to use a batch size which is a multiple of the total number of examples in the test set. Here are the dataset sizes:
  - CUB: 5794
  - Oxford Flowers: 6149
  - PartImageNet OOD: 1658
  - PartImageNet Seg: 2405
  - NABirds: 24633
- We provide a function called [factors](utils/misc_utils.py) which can be used to find the factors of these dataset sizes.
  
## Classification
- For classification evaluation, simply adapt the command from [training instructions](training_instructions.md) by adding the `--eval_only` flag. 
- The command should look like this:
  ```
  python train_net.py \
  --eval_only \
  --snapshot_dir <path to the model checkpoint> \
  --dataset <dataset name> \
  <other required arguments>
  ```
- There is no need to specify the `--wandb` flag for evaluation. All the metrics will be printed to the console.

## Part Discovery
- For part discovery evaluation, use the following command:
  ```
  python evaluate_parts.py \
  --model_path <path to the model checkpoint> \
  --dataset <dataset name> \
  --center_crop \
  --eval_mode <mode> \
  --num_parts <number of foreground parts predicted by the model> \
  <model specific arguments> 
  <dataset specific arguments>
  ```
### Specific Arguments
- `--eval_mode`: There are 3 options: `nmi_ari`, `keypoint`, `fg_bg`.
  - `nmi_ari`: This mode evaluates the model's part discovery performance using the Normalized Mutual Information (NMI) and Adjusted Rand Index (ARI) metrics. This mode is used for CUB, NABirds, PartImageNet OOD and PartImageNetSeg datasets.
  - `keypoint`: This mode evaluates the model's part discovery performance using the keypoint detection metrics. This mode is used for CUB and NABirds datasets.
  - `fg_bg`: This mode evaluates the model's part discovery performance using the foreground-background segmentation metrics. This mode is used only for Oxford Flowers dataset.
- `--num_parts`: The number of foreground parts predicted by the model. This is the same value that was used during training.
- `--center_crop`: This flag is necessary for evaluation on Vision Transformers. It crops the center of the image to the required size before evaluation. This is necessary because the Vision Transformer model requires a fixed input size. Additionally, if you want to evaluate with batch size > 1, you need to use the `--center_crop` flag.
- `--model_path`: The path to the model checkpoint.
- `--dataset`: The name of the dataset. This is used to load the dataset and the corresponding evaluation metrics. The options are: `cub`, `part_imagenet` and `flowers102seg`. Note: For NABirds, use `cub` as the dataset name. As the dataset is similar to CUB, the evaluation metrics and dataset loading functions are the same.

