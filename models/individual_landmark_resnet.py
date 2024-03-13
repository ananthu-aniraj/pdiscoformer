# Modified from https://github.com/robertdvdk/part_detection/blob/main/nets.py
import torch
from torch import Tensor
from torch.nn import Softmax2d, Parameter
from typing import Any
from .layers.independent_mlp import IndependentMLPs


# Baseline model, a modified ResNet with reduced downsampling for a spatially larger feature tensor in the last layer
class IndividualLandmarkResNet(torch.nn.Module):
    def __init__(self, init_model: torch.nn.Module, num_landmarks: int = 8,
                 num_classes: int = 200, sl_channels: int = 1024, fl_channels: int = 2048,
                 use_torchvision_model: bool = False, part_dropout: float = 0.3,
                 modulation_type: str = "original", modulation_orth: bool = False, gumbel_softmax: bool = False,
                 gumbel_softmax_temperature: float = 1.0, gumbel_softmax_hard: bool = False,
                 classifier_type: str = "linear", noise_variance: float = 0.0) -> None:
        super().__init__()

        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.noise_variance = noise_variance
        self.conv1 = init_model.conv1
        self.bn1 = init_model.bn1
        if use_torchvision_model:
            self.act1 = init_model.relu
        else:
            self.act1 = init_model.act1
        self.maxpool = init_model.maxpool
        self.layer1 = init_model.layer1
        self.layer2 = init_model.layer2
        self.layer3 = init_model.layer3
        self.layer4 = init_model.layer4
        self.feature_dim = sl_channels + fl_channels
        self.fc_landmarks = torch.nn.Conv2d(self.feature_dim, num_landmarks + 1, 1, bias=False)
        self.gumbel_softmax = gumbel_softmax
        self.gumbel_softmax_temperature = gumbel_softmax_temperature
        self.gumbel_softmax_hard = gumbel_softmax_hard
        self.modulation_type = modulation_type
        if modulation_type == "layer_norm":
            self.modulation = torch.nn.LayerNorm([self.feature_dim, self.num_landmarks + 1])
        elif modulation_type == "original":
            self.modulation = torch.nn.Parameter(torch.ones(1, self.feature_dim, self.num_landmarks + 1))
        elif modulation_type == "parallel_mlp":
            self.modulation = IndependentMLPs(part_dim=self.num_landmarks + 1, latent_dim=self.feature_dim,
                                              num_lin_layers=1, act_layer=True, bias=True)
        elif modulation_type == "parallel_mlp_no_bias":
            self.modulation = IndependentMLPs(part_dim=self.num_landmarks + 1, latent_dim=self.feature_dim,
                                              num_lin_layers=1, act_layer=True, bias=False)
        elif modulation_type == "parallel_mlp_no_act":
            self.modulation = IndependentMLPs(part_dim=self.num_landmarks + 1, latent_dim=self.feature_dim,
                                              num_lin_layers=1, act_layer=False, bias=True)
        elif modulation_type == "parallel_mlp_no_act_no_bias":
            self.modulation = IndependentMLPs(part_dim=self.num_landmarks + 1, latent_dim=self.feature_dim,
                                              num_lin_layers=1, act_layer=False, bias=False)
        elif modulation_type == "none":
            self.modulation = torch.nn.Identity()
        else:
            raise ValueError("modulation_type not implemented")

        self.modulation_orth = modulation_orth

        self.dropout_full_landmarks = torch.nn.Dropout1d(part_dropout)
        self.classifier_type = classifier_type
        if classifier_type == "independent_mlp":
            self.fc_class_landmarks = IndependentMLPs(part_dim=self.num_landmarks, latent_dim=self.feature_dim,
                                                      num_lin_layers=1, act_layer=False, out_dim=num_classes,
                                                      bias=False, stack_dim=1)
        elif classifier_type == "linear":
            self.fc_class_landmarks = torch.nn.Linear(in_features=self.feature_dim, out_features=num_classes,
                                                      bias=False)
        else:
            raise ValueError("classifier_type not implemented")

    def forward(self, x: Tensor) -> tuple[Any, Any, Any, Any, Parameter, int | Any]:
        # Pretrained ResNet part of the model
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x = self.layer4(l3)
        x = torch.nn.functional.interpolate(x, size=(l3.shape[-2], l3.shape[-1]), mode='bilinear', align_corners=False)
        x = torch.cat((x, l3), dim=1)

        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps resnet, a = convolution kernel
        batch_size = x.shape[0]

        ab = self.fc_landmarks(x)
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1).contiguous()
        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, x.shape[-2],
                                                                          x.shape[-1]).contiguous()
        a_sq = a_sq.permute(1, 0, 2, 3).contiguous()

        dist = b_sq - 2 * ab + a_sq
        maps = -dist

        # Softmax so that the attention maps for each pixel add up to 1
        if self.gumbel_softmax:
            maps = torch.nn.functional.gumbel_softmax(maps, dim=1, tau=self.gumbel_softmax_temperature,
                                                      hard=self.gumbel_softmax_hard)  # [B, num_landmarks + 1, H, W]
        else:
            maps = torch.nn.functional.softmax(maps, dim=1)  # [B, num_landmarks + 1, H, W]

        # Use maps to get weighted average features per landmark
        all_features = (maps.unsqueeze(1) * x.unsqueeze(2)).mean(-1).mean(-1).contiguous()
        if self.noise_variance > 0.0:
            all_features += torch.randn_like(all_features,
                                             device=all_features.device) * x.std().detach() * self.noise_variance

        # Modulate the features
        if self.modulation_type == "original":
            all_features_mod = all_features * self.modulation
        else:
            all_features_mod = self.modulation(all_features)

        # Classification based on the landmark features
        scores = self.fc_class_landmarks(
            self.dropout_full_landmarks(all_features_mod[..., :-1].permute(0, 2, 1).contiguous())).permute(0, 2,
                                                                                                           1).contiguous()
        if self.modulation_orth:
            return all_features_mod, maps, scores, dist
        else:
            return all_features, maps, scores, dist
