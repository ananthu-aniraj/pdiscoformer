# Attention Block with option to return the mean of k over heads from attention

import torch
from timm.models.vision_transformer import Attention, Block
import torch.nn.functional as F
from typing import Tuple


class AttentionWQKVReturn(Attention):
    """
    Modifications:
         - Return the qkv tensors from the attention
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, torch.stack((q, k, v), dim=0)


class BlockWQKVReturn(Block):
    """
    Modifications:
        - Use AttentionWQKVReturn instead of Attention
        - Return the qkv tensors from the attention
    """

    def forward(self, x: torch.Tensor, return_qkv: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        x_attn, qkv = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(x_attn))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        if return_qkv:
            return x, qkv
        else:
            return x
