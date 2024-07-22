import torch
import os
import copy
from pathlib import Path
from models.individual_landmark_resnet import pdisconet_resnet_torchvision_bb
from models.individual_landmark_vit import pdiscoformer_vit_bb, pdisconet_vit_bb
from utils.training_utils.engine_utils import load_state_dict_pdisco

dependencies = ['torch', 'torchvision', 'timm']
base_release_url = "https://github.com/ananthu-aniraj/pdiscoformer/releases/download/"
cub_base_url = base_release_url + "pdiscoformer_cub_weights/"
flowers_base_url = base_release_url + "pdiscoformer_flowers_weights/"
part_imagenet_ood_base_url = base_release_url + "pdiscormer_partimagenet_ood_weights/"
part_imagenet_seg_base_url = base_release_url + "pdiscoformer_partimagent_seg_weights/"
nabirds_base_url = base_release_url + "pdiscoformer_nabirds_weights/"
nabirds_pdisconet_resnet_base_url = base_release_url + "pdisconet_resnet_nabirds_weights/"
nabirds_pdisconet_vit_base_url = base_release_url + "pdisconet_vit_nabirds_weights/"
pretrained_k_values = {
    "cub": [4, 8, 16],
    "flowers": [2, 4, 8],
    "part_imagenet_ood": [8, 25, 50],
    "part_imagenet_seg": [8, 16, 25, 41, 50],
    "nabirds": [4, 8, 11]
}
pretrained_image_size = {
    "cub": 518,
    "flowers": 224,
    "part_imagenet_ood": 224,
    "part_imagenet_seg": 224,
    "nabirds": 518
}
num_classes = {
    "cub": 200,
    "flowers": 102,
    "part_imagenet_ood": 109,
    "part_imagenet_seg": 158,
    "nabirds": 555
}
model_dataset_urls = {
    "cub": cub_base_url,
    "flowers": flowers_base_url,
    "part_imagenet_ood": part_imagenet_ood_base_url,
    "part_imagenet_seg": part_imagenet_seg_base_url,
    "nabirds": nabirds_base_url
}
supported_datasets = ["cub", "flowers", "part_imagenet_ood", "part_imagenet_seg", "nabirds"]


def pdiscoformer_vit(pretrained=True, model_dataset="cub", k=8):
    """
    Function to load the PDiscoFormer model with ViT backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :return: PDiscoFormer model with ViT backbone
    """
    if model_dataset not in supported_datasets:
        raise ValueError(f"Model dataset {model_dataset} not recognized")
    model_url = model_dataset_urls[model_dataset]
    img_size = pretrained_image_size[model_dataset]
    num_cls = num_classes[model_dataset]

    model = pdiscoformer_vit_bb("vit_base_patch14_reg4_dinov2.lvd142m", num_cls=num_cls, k=k, img_size=img_size)
    if pretrained:
        if k not in pretrained_k_values[model_dataset]:
            raise ValueError(f"Model not trained for k = {k} for dataset {model_dataset}")
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "pdiscoformer_checkpoints", f"pdiscoformer_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_pdisco(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model


def pdisconet_vit(pretrained=True, model_dataset="nabirds", k=8):
    """
    Function to load the PDiscoNet model with ViT backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :return: PDiscoNet model with ViT backbone
    """
    if "nabirds" not in model_dataset:
        raise ValueError(f"Model dataset {model_dataset} not recognized")

    model_url = nabirds_pdisconet_vit_base_url
    img_size = pretrained_image_size[model_dataset]
    num_cls = num_classes[model_dataset]

    model = pdisconet_vit_bb("vit_base_patch14_reg4_dinov2.lvd142m", num_cls=num_cls, k=k, img_size=img_size)
    if pretrained:
        if k not in pretrained_k_values[model_dataset]:
            raise ValueError(f"Model not trained for k = {k} for dataset {model_dataset}")
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "pdiscoformer_checkpoints", f"pdisconet_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_pdisco(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model


def pdisconet_resnet101(pretrained=True, model_dataset="nabirds", k=8):
    """
    Function to load the PDiscoNet model with ResNet-101 backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :return: PDiscoNet model with ResNet-101 backbone
    """
    if "nabirds" not in model_dataset:
        raise ValueError(f"Model dataset {model_dataset} not recognized")

    model_url = nabirds_pdisconet_resnet_base_url
    num_cls = num_classes[model_dataset]

    model = pdisconet_resnet_torchvision_bb("resnet101", num_cls=num_cls, k=k)
    if pretrained:
        if k not in pretrained_k_values[model_dataset]:
            raise ValueError(f"Model not trained for k = {k} for dataset {model_dataset}")
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "pdiscoformer_checkpoints", f"pdisconet_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_pdisco(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model


def _make_pdisco_model(pretrained=True, backbone="pdiscoformer_vit", model_dataset="cub", k=8, device="cpu"):
    """
    Function to load the PDiscoFormer model
    :param pretrained: Boolean flag to load the pretrained weights
    :param backbone: Type of model to load
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :param device: Device to load the model on
    """
    if backbone == "pdiscoformer_vit":
        model = pdiscoformer_vit(pretrained=pretrained, model_dataset=model_dataset, k=k)
    elif backbone == "pdisconet_vit":
        model = pdisconet_vit(pretrained=pretrained, model_dataset=model_dataset, k=k)
    elif backbone == "pdisconet_resnet101":
        model = pdisconet_resnet101(pretrained=pretrained, model_dataset=model_dataset, k=k)
    else:
        raise ValueError(f"Model type {backbone} not recognized")
    model = model.to(device)
    return model
