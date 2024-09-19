# This file contains the implementation of the IndependentMLPs class
import torch


class IndependentMLPs(torch.nn.Module):
    """
    This class implements the MLP used for classification with the option to use an additional independent MLP layer
    """

    def __init__(self, part_dim, latent_dim, bias=False, num_lin_layers=1, act_layer=True, out_dim=None, stack_dim=-1):
        """

        :param part_dim: Number of parts
        :param latent_dim: Latent dimension
        :param bias: Whether to use bias
        :param num_lin_layers: Number of linear layers
        :param act_layer: Whether to use activation layer
        :param out_dim: Output dimension (default: None)
        :param stack_dim: Dimension to stack the outputs (default: -1)
        """

        super().__init__()

        self.bias = bias
        self.latent_dim = latent_dim
        if out_dim is None:
            out_dim = latent_dim
        self.out_dim = out_dim
        self.part_dim = part_dim
        self.stack_dim = stack_dim

        layer_stack = torch.nn.ModuleList()
        for i in range(part_dim):
            layer_stack.append(torch.nn.Sequential())
            for j in range(num_lin_layers):
                layer_stack[i].add_module(f"fc_{j}", torch.nn.Linear(latent_dim, self.out_dim, bias=bias))
                if act_layer:
                    layer_stack[i].add_module(f"act_{j}", torch.nn.GELU())
        self.feature_layers = layer_stack
        self.reset_weights()

    def __repr__(self):
        return f"IndependentMLPs(part_dim={self.part_dim}, latent_dim={self.latent_dim}), bias={self.bias}"

    def reset_weights(self):
        """ Initialize weights with a identity matrix"""
        for layer in self.feature_layers:
            for m in layer.modules():
                if isinstance(m, torch.nn.Linear):
                    # Initialize weights with a truncated normal distribution
                    torch.nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """ Input X has the dimensions batch x latent_dim x part_dim """

        outputs = []
        for i, layer in enumerate(self.feature_layers):
            if self.stack_dim == -1:
                in_ = x[..., i]
            else:
                in_ = x[:, i, ...]  # Select feature i
            out = layer(in_)  # Apply MLP to feature i
            outputs.append(out)

        x = torch.stack(outputs, dim=self.stack_dim)  # Stack the outputs

        return x
