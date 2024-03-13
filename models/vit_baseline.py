# Compostion of the VisionTransformer class from timm.models.vision_transformer with extra features: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
import torch
import torch.nn as nn
from typing import Tuple, Union, Sequence, Any
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block, Attention
from .layers.transformer_layers import BlockWQKVReturn, AttentionWQKVReturn

from utils.misc_utils import compute_attention, compute_dot_product_similarity


class BaselineViT(torch.nn.Module):
    """
    Modifications:
    - Use PDiscoBlock instead of Block
    - Use PDiscoAttention instead of Attention
    - Return the mean of k over heads from attention
    - Option to use only class tokens or only patch tokens or both (concat) for classification
    """

    def __init__(self, init_model: torch.nn.Module, num_classes: int,
                 class_tokens_only: bool = False,
                 patch_tokens_only: bool = False, return_transformer_qkv: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_tokens_only = class_tokens_only
        self.patch_tokens_only = patch_tokens_only
        self.num_prefix_tokens = init_model.num_prefix_tokens
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        self.cls_token = init_model.cls_token
        self.reg_token = init_model.reg_token

        self.patch_embed = init_model.patch_embed

        self.pos_embed = init_model.pos_embed
        self.pos_drop = init_model.pos_drop
        self.part_embed = nn.Identity()
        self.patch_prune = nn.Identity()
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks
        self.norm = init_model.norm

        self.fc_norm = init_model.fc_norm
        if class_tokens_only or patch_tokens_only:
            self.head = nn.Linear(init_model.embed_dim, num_classes)
        else:
            self.head = nn.Linear(init_model.embed_dim * 2, num_classes)

        self.h_fmap = int(self.patch_embed.img_size[0] // self.patch_embed.patch_size[0])
        self.w_fmap = int(self.patch_embed.img_size[1] // self.patch_embed.patch_size[1])

        self.return_transformer_qkv = return_transformer_qkv
        self.convert_blocks_and_attention()
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

    def _init_weights_head(self):
        trunc_normal_(self.head.weight, std=.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        x = self.patch_embed(x)

        # Position Embedding
        x = self._pos_embed(x)

        x = self.part_embed(x)
        x = self.patch_prune(x)

        # Forward pass through transformer
        x = self.norm_pre(x)

        if self.return_transformer_qkv:
            # Return keys of last attention layer
            for i, blk in enumerate(self.blocks):
                x, qkv = blk(x, return_qkv=True)
        else:
            x = self.blocks(x)

        x = self.norm(x)

        # Classification head
        x = self.fc_norm(x)
        if self.class_tokens_only:  # only use class token
            x = x[:, 0, :]
        elif self.patch_tokens_only:  # only use patch tokens
            x = x[:, self.num_prefix_tokens:, :].mean(dim=1)
        else:
            x = torch.cat([x[:, 0, :], x[:, self.num_prefix_tokens:, :].mean(dim=1)], dim=1)
        x = self.head(x)
        if self.return_transformer_qkv:
            return x, qkv
        else:
            return x

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
