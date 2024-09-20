# Compostion of the VisionTransformer class from timm with extra features: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
from pathlib import Path
import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Union, Sequence, Optional, Dict

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, EntryNotFoundError, constants
from timm.models import create_model
from timm.models.vision_transformer import Block, Attention
from utils.misc_utils import compute_attention

from layers.transformer_layers import BlockWQKVReturn, AttentionWQKVReturn
from layers.independent_mlp import IndependentMLPs


class IndividualLandmarkViT(torch.nn.Module, PyTorchModelHubMixin,
                            pipeline_tag='image-classification',
                            repo_url='https://github.com/ananthu-aniraj/pdiscoformer'):

    def __init__(self, init_model: torch.nn.Module, num_landmarks: int = 8, num_classes: int = 200,
                 part_dropout: float = 0.3, return_transformer_qkv: bool = False,
                 modulation_type: str = "original", gumbel_softmax: bool = False,
                 gumbel_softmax_temperature: float = 1.0, gumbel_softmax_hard: bool = False,
                 modulation_orth: bool = False, classifier_type: str = "linear", noise_variance: float = 0.0) -> None:
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.noise_variance = noise_variance
        self.num_prefix_tokens = init_model.num_prefix_tokens
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        self.cls_token = init_model.cls_token
        self.reg_token = init_model.reg_token

        self.feature_dim = init_model.embed_dim
        self.patch_embed = init_model.patch_embed
        self.pos_embed = init_model.pos_embed
        self.pos_drop = init_model.pos_drop
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks
        self.norm = init_model.norm
        self.return_transformer_qkv = return_transformer_qkv
        self.h_fmap = int(self.patch_embed.img_size[0] // self.patch_embed.patch_size[0])
        self.w_fmap = int(self.patch_embed.img_size[1] // self.patch_embed.patch_size[1])

        self.unflatten = nn.Unflatten(1, (self.h_fmap, self.w_fmap))
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
        self.convert_blocks_and_attention()
        self._init_weights()

    def _init_weights_head(self):
        # Initialize weights with a truncated normal distribution
        if self.classifier_type == "independent_mlp":
            self.fc_class_landmarks.reset_weights()
        else:
            torch.nn.init.trunc_normal_(self.fc_class_landmarks.weight, std=0.02)
            if self.fc_class_landmarks.bias is not None:
                torch.nn.init.zeros_(self.fc_class_landmarks.bias)

    def _init_weights(self):
        self._init_weights_head()

    def convert_blocks_and_attention(self):
        for module in self.modules():
            if isinstance(module, Block):
                module.__class__ = BlockWQKVReturn
            elif isinstance(module, Attention):
                module.__class__ = AttentionWQKVReturn

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def forward(self, x: Tensor) -> tuple[Any, Any, Any, Any, int | Any] | tuple[Any, Any, Any, Any, int | Any]:

        x = self.patch_embed(x)

        # Position Embedding
        x = self._pos_embed(x)

        # Forward pass through transformer
        x = self.norm_pre(x)

        x = self.blocks(x)
        x = self.norm(x)

        # Compute per landmark attention maps
        # (b - a)^2 = b^2 - 2ab + a^2, b = feature maps vit, a = convolution kernel
        batch_size = x.shape[0]
        x = x[:, self.num_prefix_tokens:, :]  # [B, num_patch_tokens, embed_dim]
        x = self.unflatten(x)  # [B, H, W, embed_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, H, W]
        ab = self.fc_landmarks(x)  # [B, num_landmarks + 1, H, W]
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_landmarks + 1, -1, -1).contiguous()
        a_sq = self.fc_landmarks.weight.pow(2).sum(1, keepdim=True).expand(-1, batch_size, x.shape[-2],
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
        all_features = (maps.unsqueeze(1) * x.unsqueeze(2)).contiguous()
        if self.noise_variance > 0.0:
            all_features += torch.randn_like(all_features,
                                             device=all_features.device) * x.std().detach() * self.noise_variance

        all_features = all_features.mean(-1).mean(-1).contiguous()  # [B, embed_dim, num_landmarks + 1]

        # Modulate the features
        if self.modulation_type == "original":
            all_features_mod = all_features * self.modulation  # [B, embed_dim, num_landmarks + 1]
        else:
            all_features_mod = self.modulation(all_features)  # [B, embed_dim, num_landmarks + 1]

        # Classification based on the landmark features
        scores = self.fc_class_landmarks(
            self.dropout_full_landmarks(all_features_mod[..., :-1].permute(0, 2, 1).contiguous())).permute(0, 2,
                                                                                                           1).contiguous()
        if self.modulation_orth:
            return all_features_mod, maps, scores, dist
        else:
            return all_features, maps, scores, dist

    def get_specific_intermediate_layer(
            self,
            x: torch.Tensor,
            n: int = 1,
            return_qkv: bool = False,
            return_att_weights: bool = False,
    ):
        num_blocks = len(self.blocks)
        attn_weights = []
        if n >= num_blocks:
            raise ValueError(f"n must be less than {num_blocks}")

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        if n == -1:
            if return_qkv:
                raise ValueError("take_indice cannot be -1 if return_transformer_qkv is True")
            else:
                return x

        for i, blk in enumerate(self.blocks):
            if self.return_transformer_qkv:
                x, qkv = blk(x, return_qkv=True)

                if return_att_weights:
                    attn_weight, _ = compute_attention(qkv)
                    attn_weights.append(attn_weight.detach())
            else:
                x = blk(x)
            if i == n:
                output = x.clone()
                if self.return_transformer_qkv and return_qkv:
                    qkv_output = qkv.clone()
                break
        if self.return_transformer_qkv and return_qkv and return_att_weights:
            return output, qkv_output, attn_weights
        elif self.return_transformer_qkv and return_qkv:
            return output, qkv_output
        elif self.return_transformer_qkv and return_att_weights:
            return output, attn_weights
        else:
            return output

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        if self.return_transformer_qkv:
            qkv_outputs = []
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        for i, blk in enumerate(self.blocks):
            if self.return_transformer_qkv:
                x, qkv = blk(x, return_qkv=True)
            else:
                x = blk(x)
            if i in take_indices:
                outputs.append(x)
                if self.return_transformer_qkv:
                    qkv_outputs.append(qkv)
        if self.return_transformer_qkv:
            return outputs, qkv_outputs
        else:
            return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> tuple[tuple, Any]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        if self.return_transformer_qkv:
            outputs, qkv = self._intermediate_layers(x, n)
        else:
            outputs = self._intermediate_layers(x, n)

        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return_out = tuple(zip(outputs, prefix_tokens))
        else:
            return_out = tuple(outputs)

        if self.return_transformer_qkv:
            return return_out, qkv
        else:
            return return_out

    @classmethod
    def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: Optional[str],
            cache_dir: Optional[Union[str, Path]],
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: Optional[bool],
            local_files_only: bool,
            token: Union[str, bool, None],
            map_location: str = "cpu",
            strict: bool = False,
            timm_backbone: str = "hf_hub:timm/vit_base_patch14_reg4_dinov2.lvd142m",
            input_size: int = 518,
            **model_kwargs):
        base_model = create_model(timm_backbone, pretrained=True, img_size=input_size)
        model = cls(base_model, **model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=constants.PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)


def pdiscoformer_vit_bb(backbone, img_size=224, num_cls=200, k=8, **kwargs):
    base_model = create_model(
        backbone,
        pretrained=False,
        img_size=img_size,
    )

    model = IndividualLandmarkViT(base_model, num_landmarks=k, num_classes=num_cls,
                                  modulation_type="layer_norm", gumbel_softmax=True,
                                  modulation_orth=True)
    return model


def pdisconet_vit_bb(backbone, img_size=224, num_cls=200, k=8, **kwargs):
    base_model = create_model(
        backbone,
        pretrained=False,
        img_size=img_size,
    )

    model = IndividualLandmarkViT(base_model, num_landmarks=k, num_classes=num_cls,
                                  modulation_type="original")
    return model
