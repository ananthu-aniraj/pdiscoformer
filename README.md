# PDiscoFormer: Relaxing Part Discovery Constraints with Vision Transformers 
Official implementation of the paper "PDiscoFormer: Relaxing Part Discovery Constraints with Vision Transformers", accepted at ECCV 2024. 


[[`Arxiv`]](https://arxiv.org/abs/2407.04538)

## Introduction
In our work, we explore computer vision methods that perform unsupervised part discovery. We introduce a novel method and training objective for this task using self-supervised vision transformers, achieving state-of-the-art results. Our model learns to discover consistent, discriminative parts that are useful for solving image classification tasks, taking a step towards inherently interpretable models.



![splash_pdisconetV2 (1)](https://github.com/ananthu-aniraj/pdiscoformer/assets/50333505/aa3803c0-2ce0-411e-bb04-a79113c9da07)

![PosterPdiscoformer drawio (1)](https://github.com/ananthu-aniraj/pdiscoformer/assets/50333505/6f8b3453-a4fe-4eda-9e81-741ef3420687)



# Abstract
Computer vision methods that explicitly detect object parts and reason on them are a step towards inherently interpretable models. Existing approaches that perform part discovery driven by a fine-grained classification task make very restrictive assumptions on the geometric properties of the discovered parts; they should be small and compact. Although this prior is useful in some cases, in this paper we show that pre-trained transformer-based vision models, such as self-supervised DINOv2 ViT, enable the relaxation of these constraints. In particular, we find that a total variation (TV) prior, which allows for multiple connected components of any size, substantially outperforms previous work. We test our approach on three fine-grained classification benchmarks: CUB, PartImageNet and Oxford Flowers, and compare our results to previously published methods as well as a re-implementation of the state-of-the-art method PDiscoNet with a transformer-based backbone. We consistently obtain substantial improvements across the board, both on part discovery metrics and the downstream classification task, showing that the strong inductive biases in self-supervised ViT models require to rethink the geometric priors that can be used for unsupervised part discovery.


# Model Architecture
![image](https://github.com/ananthu-aniraj/pdiscoformer/assets/50333505/73c30fb1-2f2c-408a-81dd-4447f9091f86)



# Setup
To install the required packages, run the following command:
```conda env create -f environment.yml```

Otherwise, you can also individually install the following packages:
1. [PyTorch](https://pytorch.org/get-started/locally/): Tested upto version 2.3, please raise an issue if you face any problems with more recent versions.
2. [Colorcet](https://colorcet.holoviz.org/getting_started/index.html)
3. [Matplotlib](https://matplotlib.org/stable/users/installing.html)
3. [OpenCV](https://pypi.org/project/opencv-python-headless/)
4. [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
5. [Scikit-Image](https://scikit-image.org/docs/stable/install.html)
6. [Scikit-Learn](https://scikit-learn.org/stable/install.html) 
7. [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/pages/install.html)
8. [timm](https://pypi.org/project/timm/)
9. [wandb](https://pypi.org/project/wandb/): It is recommended to create an account and use it for tracking the experiments. Use the '--wandb' flag when running the training script to enable this feature.
10. [pycocotools](https://pypi.org/project/pycocotools/)
11. [pytopk](https://pypi.org/project/pytopk/)

# Update
The code has been updated to support the NABirds dataset. The corresponding evaluation metrics and pre-trained models have also been added.

# Datasets
### CUB
The dataset can be downloaded from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/). 

The folder structure should look like this:

```
CUB_200_2011
├── attributes
├── bounding_boxes.txt
├── classes.txt
├── images
├── image_class_labels.txt
├── images.txt
├── parts
├── README
└── train_test_split.txt
```

### PartImageNet OOD
The dataset can be downloaded from [here](https://github.com/TACJu/PartImageNet).
After downloading the dataset, use the pre-processing script (prepare_partimagenet_ood.py) and train-test split (data_sets/train_test_split_pimagenet_ood.txt) to generate the required annotation files for training and evaluation.
The command to run the pre-processing script is as follows:

```python prepare_partimagenet_ood.py --anno_path <path to train.json file> --output_dir <path to save the train and test json file> --train_test_split_file data_sets/train_test_split_pimagenet_ood.txt```

### Oxford Flowers
The dataset is automatically downloaded by the training script with the required folder structure (except for the segmentation masks).
If you want to evaluate the foreground segmentation on the dataset, please download the segmentations from [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).
The final folder structure should look like this:

```
(root folder)
├── flowers-102 (folder containing the dataset created automatically by the training script)
    ├── segmim (folder containing the segmentation masks)
    ├── jpg
    ├── imagelabels.mat
    └── setid.mat
```
### PartImageNet Seg
The dataset can be downloaded from [here](https://github.com/TACJu/PartImageNet). No additional pre-processing is required.

### NABirds
The dataset can be downloaded from [here](https://dl.allaboutbirds.org/nabirds). 
The experiments on this dataset are not present in the paper as they were conducted after the paper was submitted. 
The folder structure should look like this (essentially the same as CUB except for the attributes):

```
nabirds
├── bounding_boxes.txt
├── classes.txt
├── images
├── image_class_labels.txt
├── images.txt
├── parts
├── hierarchy.txt
├── README
└── train_test_split.txt
```

# Training
The details of running the training script can be found in the [training instructions](training_instructions.md) file.

# Evaluation
The details of running the evaluation metrics for both classification and part discovery can be found in the [evaluation instructions](evaluation_instructions.md) file.

# Model Zoo
The trained models can be found in the [model zoo](model_zoo.md) file. 


# Issues and Questions
Feel free to raise an issue if you face any problems with the code or have any questions about the paper.



