# Model Zoo

We provide the pre-trained models for the following datasets:
- CUB-200-2011
- PartImageNet OOD
- Oxford Flowers
- PartImageNet Seg
- NABirds

The models can be downloaded from the links provided below. They can also be loaded using torch hub using the code snippets provided below.

## How to Get Started with the Model with Hugging Face 🤗
```python
from models import IndividualLandmarkViT

# CUB Models
pdiscoformer_cub_k_4 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_cub_k_4")
pdiscoformer_cub_k_8 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_cub_k_8")
pdiscoformer_cub_k_16 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_cub_k_16")

# PartImageNet OOD Models
pdiscoformer_partimagenet_ood_k_8 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_part_imagenet_ood_k_8", input_size=224)
pdiscoformer_partimagenet_ood_k_25 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_part_imagenet_ood_k_25", input_size=224)
pdiscoformer_partimagenet_ood_k_50 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_part_imagenet_ood_k_50", input_size=224)

# Oxford Flowers Models
pdiscoformer_flowers_k_2 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_flowers_k_2", input_size=224)
pdiscoformer_flowers_k_4 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_flowers_k_4", input_size=224)
pdiscoformer_flowers_k_8 = IndividualLandmarkViT.from_pretrained("ananthu-aniraj/pdiscoformer_flowers_k_8", input_size=224)
```



## How to Get Started with the Model with Torch Hub

```python
import torch

# CUB Models
pdiscoformer_cub_k_4 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_cub_k_4', pretrained=True, trust_repo=True)
pdiscoformer_cub_k_8 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_cub_k_8', pretrained=True, trust_repo=True)
pdiscoformer_cub_k_16 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_cub_k_16', pretrained=True, trust_repo=True)

# PartImageNet OOD Models
pdiscoformer_partimagenet_ood_k_8 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_pimagenet_k_8', pretrained=True, trust_repo=True)
pdiscoformer_partimagenet_ood_k_25 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_pimagenet_k_25', pretrained=True, trust_repo=True)
pdiscoformer_partimagenet_ood_k_50 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_pimagenet_k_50', pretrained=True, trust_repo=True)


# Oxford Flowers Models
pdiscoformer_flowers_k_2 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_flowers_k_2', pretrained=True, trust_repo=True)
pdiscoformer_flowers_k_4 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_flowers_k_4', pretrained=True, trust_repo=True)
pdiscoformer_flowers_k_8 = torch.hub.load("ananthu-aniraj/pdiscoformer:main", 'pdiscoformer_flowers_k_8', pretrained=True, trust_repo=True)
```

The full list of model keys are provided using the following code snippet:

```python 
import torch
torch.hub.list("ananthu-aniraj/pdiscoformer:main")
```

# Pre-trained Models
Please note that these models were recently trained and may have slight deviations in performance compared to the models reported in the paper.

## CUB-200-2011

<table>
    <tr>
        <th>Model</th>
        <th>Backbone</th>
        <th>K</th>
        <th>URL</th>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>4</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_cub_weights/4_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>        
        <td>8</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_cub_weights/8_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>        
        <td>16</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_cub_weights/16_parts_snapshot_best.pt">Download</a></td>
    </tr>
</table>

## PartImageNet OOD

<table>
    <tr>
        <th>Model</th>
        <th>Backbone</th>
        <th>K</th>
        <th>URL</th>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>8</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscormer_partimagenet_ood_weights/8_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>25</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscormer_partimagenet_ood_weights/25_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>50</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscormer_partimagenet_ood_weights/50_parts_snapshot_best.pt">Download</a></td>
    </tr>
</table>

## Oxford Flowers

<table>
    <tr>
        <th>Model</th>
        <th>Backbone</th>
        <th>K</th>
        <th>URL</th>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>2</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_flowers_weights/2_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>4</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_flowers_weights/4_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>8</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_flowers_weights/8_parts_snapshot_best.pt">Download</a></td>
    </tr>
</table>

## PartImageNet Seg

<table>
    <tr>
        <th>Model</th>
        <th>Backbone</th>
        <th>K</th>
        <th>URL</th>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>8</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_partimagent_seg_weights/8_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>16</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_partimagent_seg_weights/16_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>25</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_partimagent_seg_weights/25_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>41</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_partimagent_seg_weights/41_parts_snapshot_best.pt">Download</a></td>
    </tr>
    <tr>
        <td>PdiscoFormer</td>
        <td><a href="https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m">ViT-B</a></td>
        <td>50</td>
        <td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_partimagent_seg_weights/50_parts_snapshot_best.pt">Download</a></td>
    </tr>
</table>

## NABirds
<table><tbody>
<tr><th>Method</th><th>K</th><th>Kp</th><th>NMI</th><th>ARI</th><th>Top-1 Accuracy</th><th>URL</th></tr><tr><td>Dino</td><td>4</td><td>-</td><td>26.50</td><td>11.30</td><td>-</td><td>-</td></tr><tr><td>Dino</td><td>8</td><td>-</td><td>39.45</td><td>23.20</td><td>-</td><td>-</td></tr><tr><td>Dino</td><td>11</td><td>-</td><td>39.23</td><td>23.37</td><td>-</td><td>-</td></tr><tr><td>Huang</td><td>4</td><td>14.54</td><td>31.99</td><td>19.31</td><td>85.46</td><td>-</td></tr><tr><td>Huang</td><td>8</td><td>13.47</td><td>42.06</td><td>27.32</td><td>85.17</td><td>-</td></tr><tr><td>Huang</td><td>11</td><td>12.82</td><td>44.08</td><td>29.28</td><td>85.14</td><td>-</td></tr>
<tr><td>PDiscoNet</td><td>4</td><td>11.5</td><td>31.93</td><td>13.32</td><td>83.56</td><td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdisconet_resnet_nabirds_weights/4_parts_snapshot_best.pt">Download</a></td></tr>
<tr><td>PDiscoNet</td><td>8</td><td>11.19</td><td>37.60</td><td>19.47</td><td>84.31</td><td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdisconet_resnet_nabirds_weights/8_parts_snapshot_best.pt">Download</a></td></tr><tr>
<td>PDiscoNet</td><td>11</td><td>9.59</td><td>43.57</td><td>29.63</td><td>84.51</td><td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdisconet_resnet_nabirds_weights/11_parts_snapshot_best.pt">Download</a></td></tr>
<tr><td>PDiscoNet + ViT-B</td><td>4</td><td>9.76</td><td>43.02</td><td>22.77</td><td>87.74</td><td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdisconet_vit_nabirds_weights/4_parts_snapshot_best.pt">Download</a></td></tr>
<tr><td>PDiscoNet + ViT-B</td><td>8</td><td>9.17</td><td>56.50</td><td>34.10</td><td>85.60</td><td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdisconet_vit_nabirds_weights/8_parts_snapshot_best.pt">Download</a></td></tr><tr>
<td>PDiscoNet + ViT-B</td><td>11</td><td>9.34</td><td>68.92</td><td>54.65</td><td>83.37</td><td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdisconet_vit_nabirds_weights/11_parts_snapshot_best.pt">Download</a></td></tr><tr>
<td>PDiscoFormer</td><td>4</td><td>11.22</td><td>48.24</td><td>27.73</td><td><b>89.29</b></td><td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_nabirds_weights/4_parts_snapshot_best.pt">Download</a></td></tr><tr>
<td>PDiscoFormer</td><td>8</td><td>8.84</td><td>60.39</td><td>46.74</td><td>88.72</td><td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_nabirds_weights/8_parts_snapshot_best.pt">Download</a></td></tr><tr>
<td>PDiscoFormer</td><td>11</td><td><b>8.36</b></td><td><b>72.04</b></td><td><b>63.35</b></td>
<td>88.69</td>
<td><a href="https://github.com/ananthu-aniraj/pdiscoformer/releases/download/pdiscoformer_nabirds_weights/11_parts_snapshot_best.pt">Download</a></td></tr></tbody></table>
