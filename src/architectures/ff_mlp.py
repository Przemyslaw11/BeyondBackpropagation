# File: src/architectures/ff_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Type  # Added Type
import math
import logging

logger = logging.getLogger(__name__)


class FF_Layer(nn.Module):
    """
    A single layer for the Forward-Forward algorithm.
    Follows Linear -> Norm -> Activation pattern.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_cls: Type[nn.Module] = nn.ReLU,  # Pass class, not instance
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation_cls()  # Instantiate activation here

        # Layer Normalization: Using standard nn.LayerNorm.
        # NOTE: Hinton's paper (Sec 2.1, Footnote 5) suggests a simpler version:
        # "FF uses the simplest version of layer normalization which does not subtract the mean
        # before dividing by the length of the activity vector."
        # This could be implemented as `x / torch.linalg.norm(x, dim=1, keepdim=True)`
        # Standard LayerNorm is used here for simplicity and common practice.
        # It includes learnable affine parameters (elementwise_affine=True) by default.
        self.norm_layer = (
            nn.LayerNorm(out_features, elementwise_affine=True)
            if normalize
            else nn.Identity()
        )
        self.normalize = normalize
        logger.debug(
            f"FF_Layer: In={in_features}, Out={out_features}, Normalize={normalize}, Bias={bias}, Activation={activation_cls.__name__}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer. Applies Linear -> Norm -> Activation.
        """
        x = self.linear(x)
        # Apply normalization BEFORE activation, as per common interpretation of FF
        x = self.norm_layer(x)
        x = self.activation(x)
        return x

    def forward_with_goodness(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns the sum of squared activities (goodness).
        Goodness is calculated *after* activation.
        """
        x_out = self.forward(x)
        # Goodness: sum of squares of activations per sample in the batch
        goodness = torch.sum(x_out.pow(2), dim=1)  # Sum over the feature dimension
        return x_out, goodness


class FF_MLP(nn.Module):
    """
    Multi-Layer Perceptron specifically designed for the Forward-Forward algorithm.
    Includes layers that handle normalization and provides methods for FF training/eval.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = "relu",
        normalize_layers: bool = True,
        bias: bool = True,
    ):
        """
        Args:
            input_dim: Dimensionality of the raw input features (e.g., 784 for flattened MNIST).
            hidden_dims: List of sizes for each hidden layer.
            num_classes: Number of output classes (used for label embedding).
            activation: Name of activation function ('relu', 'tanh').
            normalize_layers: Whether to apply Layer Normalization in hidden layers.
            bias: Whether to use bias terms in linear layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.normalize_layers = normalize_layers

        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers = nn.ModuleList()
        # Input dimension to the *first* FF_Layer includes space for embedded labels
        current_dim = input_dim + num_classes

        for h_dim in hidden_dims:
            self.layers.append(
                FF_Layer(
                    current_dim,
                    h_dim,
                    activation_cls=act_cls,
                    normalize=normalize_layers,
                    bias=bias,
                )
            )
            current_dim = h_dim  # Output dimension of this layer is input for next

        logger.info(f"Initialized FF_MLP with {len(self.layers)} hidden layers.")
        layer_dims_str = " -> ".join(map(str, [input_dim + num_classes] + hidden_dims))
        logger.info(f"Layer dimensions: {layer_dims_str}")

    def prepare_ff_input(
        self, x_images: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepares input for FF by embedding labels into the image tensor.
        Assumes x_images has shape [batch_size, input_dim].
        Returns tensor of shape [batch_size, input_dim + num_classes].
        """
        batch_size = x_images.shape[0]
        device = x_images.device

        # Ensure image tensor has correct input_dim before padding
        if x_images.shape[1] != self.input_dim:
            # This should typically be handled by an input_adapter before calling this
            logger.warning(
                f"Input image tensor shape {x_images.shape} dim 1 != expected input_dim {self.input_dim}. Ensure input is flattened correctly."
            )
            # Attempt to flatten if not already flat (risky, assumes spatial dims)
            if len(x_images.shape) > 2:
                try:
                    x_images = x_images.view(batch_size, -1)
                    if x_images.shape[1] != self.input_dim:
                        raise ValueError(
                            "Flattened shape still doesn't match input_dim."
                        )
                    logger.warning("Automatically flattened input tensor.")
                except Exception as e:
                    raise ValueError(
                        f"Input image tensor has dimension {x_images.shape[1]}, expected {self.input_dim}. Auto-flatten failed: {e}"
                    ) from e
            else:
                raise ValueError(
                    f"Input image tensor has dimension {x_images.shape[1]}, expected {self.input_dim}."
                )

        # Create one-hot labels
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        # Concatenate labels and images
        # [B, num_classes] + [B, input_dim] -> [B, num_classes + input_dim]
        ff_input = torch.cat((one_hot_labels, x_images), dim=1)
        return ff_input

    def forward_upto(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Performs a forward pass through layers 0 to layer_idx (exclusive).
        Assumes input 'x' already has labels embedded.

        Args:
            x: Input tensor (shape: [batch_size, input_dim + num_classes]).
            layer_idx: Index of the layer *not* to run (runs up to layer_idx - 1).
                       layer_idx=0 returns x, layer_idx=1 returns output of self.layers[0].

        Returns:
            Output activation tensor from layer layer_idx - 1.
        """
        expected_input_dim = self.input_dim + self.num_classes
        if x.shape[1] != expected_input_dim:
            raise ValueError(
                f"Input tensor shape {x.shape} dim 1 does not match expected combined dimension {expected_input_dim}"
            )
        if layer_idx < 0 or layer_idx > len(self.layers):
            raise ValueError(
                f"layer_idx {layer_idx} out of range for {len(self.layers)} layers."
            )

        current_activation = x
        for i in range(layer_idx):  # Run layers 0, 1, ..., layer_idx-1
            current_activation = self.layers[i](current_activation)
        return current_activation

    def forward_goodness_per_layer(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass calculating and returning the 'goodness' for each hidden layer's output.
        Assumes input x already has labels embedded.

        Returns:
            List[torch.Tensor]: List containing goodness scores (sum of squared activations)
                                for each hidden layer, each tensor of shape [batch_size].
        """
        expected_input_dim = self.input_dim + self.num_classes
        if x.shape[1] != expected_input_dim:
            raise ValueError(
                f"Input tensor shape {x.shape} dim 1 does not match expected combined dimension {expected_input_dim}"
            )

        layer_goodness = []
        current_activation = x
        for layer in self.layers:
            # Use the layer's specific forward_with_goodness method
            current_activation, goodness = layer.forward_with_goodness(
                current_activation
            )
            layer_goodness.append(goodness)
        return layer_goodness


# Removed the __main__ block for cleaner architecture file
