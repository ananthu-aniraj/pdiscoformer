from load_model import pdiscoformer_vit, pdisconet_vit, pdisconet_resnet101

dependencies = ['torch', 'torchvision', 'timm']
base_release_url = "https://github.com/ananthu-aniraj/pdiscoformer/releases/download/"
cub_base_url = base_release_url + "pdiscoformer_cub_weights/"
flowers_base_url = base_release_url + "pdiscoformer_flowers_weights/"
part_imagenet_ood_base_url = base_release_url + "pdiscormer_partimagenet_ood_weights/"
part_imagenet_seg_base_url = base_release_url + "pdiscoformer_partimagent_seg_weights/"
nabirds_base_url = base_release_url + "pdiscoformer_nabirds_weights/"
nabirds_pdisconet_resnet_base_url = base_release_url + "pdisconet_resnet_nabirds_weights/"
nabirds_pdisconet_vit_base_url = base_release_url + "pdisconet_vit_nabirds_weights/"
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


def _make_pdisco_model(pretrained=True, backbone="pdiscoformer_vit", model_dataset="cub", k=8, device="cpu"):
    """
    Function to load the PDiscoFormer model
    :param pretrained: Boolean flag to load the pretrained weights
    :param backbone: Type of model to load
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :param device: Device to load the model on
    """
    img_size = pretrained_image_size[model_dataset]
    num_cls = num_classes[model_dataset]
    if backbone == "pdiscoformer_vit":
        model_url = model_dataset_urls[model_dataset]
        model = pdiscoformer_vit(pretrained=pretrained, model_dataset=model_dataset, k=k, num_cls=num_cls,
                                 img_size=img_size, model_url=model_url)
    elif backbone == "pdisconet_vit":
        model = pdisconet_vit(pretrained=pretrained, model_dataset=model_dataset, k=k, num_cls=num_cls,
                             img_size=img_size, model_url=nabirds_pdisconet_vit_base_url)
    elif backbone == "pdisconet_resnet101":
        model = pdisconet_resnet101(pretrained=pretrained, model_dataset=model_dataset, k=k, num_cls=num_cls,
                                    model_url=nabirds_pdisconet_resnet_base_url)
    else:
        raise ValueError(f"Model type {backbone} not recognized")
    model = model.to(device)
    return model


def pdiscoformer_cub_k_8(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on CUB-200-2011 dataset with k=8
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on CUB-200-2011 dataset with k=8
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="cub", k=8,
                              device=device)


def pdiscoformer_cub_k_4(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on CUB-200-2011 dataset with k=4
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on CUB-200-2011 dataset with k=4
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="cub", k=4,
                              device=device)


def pdiscoformer_cub_k_16(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on CUB-200-2011 dataset with k=16
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on CUB-200-2011 dataset with k=16
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="cub", k=16,
                              device=device)


def pdiscoformer_pimagenet_k_8(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Part-ImageNet dataset with k=8
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Part-ImageNet dataset with k=8
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="part_imagenet_ood",
                              k=8,
                              device=device)


def pdiscoformer_pimagenet_k_25(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Part-ImageNet dataset with k=25
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Part-ImageNet dataset with k=25
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="part_imagenet_ood",
                              k=25,
                              device=device)


def pdiscoformer_pimagenet_k_50(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Part-ImageNet dataset with k=50
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Part-ImageNet dataset with k=50
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="part_imagenet_ood",
                              k=50,
                              device=device)


def pdiscoformer_flowers_k_2(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Flowers-102 dataset with k=2
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Flowers-102 dataset with k=2
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="flowers", k=2,
                              device=device)


def pdiscoformer_flowers_k_4(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Flowers-102 dataset with k=4
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Flowers-102 dataset with k=4
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="flowers", k=4,
                              device=device)


def pdiscoformer_flowers_k_8(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Flowers-102 dataset with k=8
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Flowers-102 dataset with k=8
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="flowers", k=8,
                              device=device)


def pdiscoformer_pimagenet_seg_k_8(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=8
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=8
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="part_imagenet_seg",
                              k=8,
                              device=device)


def pdiscoformer_pimagenet_seg_k_16(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=16
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=16
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="part_imagenet_seg",
                              k=16,
                              device=device)


def pdiscoformer_pimagenet_seg_k_25(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=25
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=25
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="part_imagenet_seg",
                              k=25,
                              device=device)


def pdiscoformer_pimagenet_seg_k_41(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=41
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=41
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="part_imagenet_seg",
                              k=41,
                              device=device)


def pdiscoformer_pimagenet_seg_k_50(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=50
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on Part-ImageNet-Seg dataset with k=50
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="part_imagenet_seg",
                              k=50,
                              device=device)


def pdiscoformer_nabirds_k_4(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on NABirds dataset with k=4
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on NABirds dataset with k=4
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="nabirds", k=4,
                              device=device)


def pdiscoformer_nabirds_k_8(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on NABirds dataset with k=8
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on NABirds dataset with k=8
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="nabirds", k=8,
                              device=device)


def pdiscoformer_nabirds_k_11(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoFormer model trained on NABirds dataset with k=11
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoFormer model trained on NABirds dataset with k=11
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdiscoformer_vit", model_dataset="nabirds", k=11,
                              device=device)


def pdisconet_vit_nabirds_k_4(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoNet model with ViT backbone trained on NABirds dataset with k=4
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoNet model with ViT backbone trained on NABirds dataset with k=4
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdisconet_vit", model_dataset="nabirds", k=4,
                              device=device)


def pdisconet_vit_nabirds_k_8(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoNet model with ViT backbone trained on NABirds dataset with k=8
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoNet model with ViT backbone trained on NABirds dataset with k=8
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdisconet_vit", model_dataset="nabirds", k=8,
                              device=device)


def pdisconet_vit_nabirds_k_11(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoNet model with ViT backbone trained on NABirds dataset with k=11
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoNet model with ViT backbone trained on NABirds dataset with k=11
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdisconet_vit", model_dataset="nabirds", k=11,
                              device=device)


def pdisconet_resnet_nabirds_k_4(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoNet model with ResNet-101 backbone trained on NABirds dataset with k=4
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoNet model with ResNet-101 backbone trained on NABirds dataset with k=4
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdisconet_resnet101", model_dataset="nabirds", k=4,
                              device=device)


def pdisconet_resnet_nabirds_k_8(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoNet model with ResNet-101 backbone trained on NABirds dataset with k=8
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoNet model with ResNet-101 backbone trained on NABirds dataset with k=8
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdisconet_resnet101", model_dataset="nabirds", k=8,
                              device=device)


def pdisconet_resnet_nabirds_k_11(pretrained=True, device="cpu"):
    """
    Function to load the PDiscoNet model with ResNet-101 backbone trained on NABirds dataset with k=11
    :param pretrained: Boolean flag to load the pretrained weights
    :param device: Device to load the model on
    :return: PDiscoNet model with ResNet-101 backbone trained on NABirds dataset with k=11
    """
    return _make_pdisco_model(pretrained=pretrained, backbone="pdisconet_resnet101", model_dataset="nabirds", k=11,
                              device=device)
