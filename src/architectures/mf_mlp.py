"""Implements the MF_MLP model for the Mono-Forward algorithm."""

import logging
import math
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MF_MLP(nn.Module):
    """A Multi-Layer Perceptron designed for the Mono-Forward (MF) algorithm.

    This class includes standard feedforward layers (W_i) and learnable projection
    matrices (M_i). The final `self.output_layer` is primarily used by the BP
    baseline for comparison.

    Projection Matrix M_i corresponds to activation a_i.
    - M_0 uses input a_0=x.
    - M_1 uses activation a_1 (output of first hidden layer).
    - ...
    - M_L uses activation a_L (output of last hidden layer L).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        """Initializes the MF_MLP model layers and projection matrices.

        Args:
            input_dim: The dimensionality of the input features.
            hidden_dims: A list of integers specifying the size of each hidden layer.
            num_classes: The number of output classes.
            activation: The activation function to use ('relu' or 'tanh').
            bias: Whether to use a bias term in the linear layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.num_hidden_layers = len(hidden_dims)

        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # --- Standard Feedforward Layers (W_i) ---
        # W_1, W_2, ..., W_L
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for _, h_dim in enumerate(hidden_dims):
            linear_layer = nn.Linear(current_dim, h_dim, bias=bias)  # Layer W_{i+1}
            if activation.lower() == "relu":
                nn.init.kaiming_uniform_(
                    linear_layer.weight, a=math.sqrt(5), nonlinearity="relu"
                )
            elif activation.lower() == "tanh":
                nn.init.xavier_uniform_(linear_layer.weight)
            else:
                nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5))

            if bias and linear_layer.bias is not None:
                nn.init.zeros_(linear_layer.bias)

            self.layers.append(linear_layer)
            self.layers.append(act_cls())
            current_dim = h_dim

        # Final classifier layer (W_L+1)
        self.output_layer = nn.Linear(current_dim, num_classes, bias=bias)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        if bias and self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

        # --- Learnable Projection Matrices (M_i) ---
        # M_i projects the activation of layer i (a_i) to goodness scores.
        # M_0 projects a_0 (input), M_1 projects a_1 (output of hidden 0), etc.
        # Need M_0 to M_L (where L = num_hidden_layers). Thus, L+1 matrices.
        self.projection_matrices = nn.ParameterList()
        # Activation dims: input_dim (a_0), hidden_dims[0] (a_1), ...,
        # hidden_dims[L-1] (a_L)
        dims_for_M = [input_dim] + hidden_dims  # Dimensions of a_0, a_1, ..., a_L
        for _, layer_dim in enumerate(dims_for_M):
            # Matrix shape is [num_classes, layer_dim] so that a_i @ M_i^T works
            m_matrix = nn.Parameter(torch.empty(num_classes, layer_dim))
            nn.init.kaiming_uniform_(m_matrix, a=math.sqrt(5))
            self.projection_matrices.append(m_matrix)

        logger.info(f"Initialized MF_MLP with {self.num_hidden_layers} hidden layers.")
        layer_dims_str = " -> ".join(
            map(str, [input_dim] + hidden_dims + [num_classes])
        )
        logger.info(f"Feedforward Layer dimensions (W): {layer_dims_str}")
        logger.info(
            "Created %d projection matrices (M_0 to M_%d).",
            len(self.projection_matrices),
            self.num_hidden_layers,
        )

    def get_projection_matrix(self, m_index: int) -> nn.Parameter:
        """Safely retrieves a projection matrix by its index.

        The index should be from 0 to num_hidden_layers.
        """
        if 0 <= m_index < len(self.projection_matrices):
            return self.projection_matrices[m_index]
        raise IndexError(
            f"Projection matrix index {m_index} out of bounds for "
            f"{len(self.projection_matrices)} matrices."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through the MLP layers, returning final logits.

        This method is used ONLY for the BP baseline evaluation. It assumes
        input 'x' is already flattened.
        """
        current_activation = x
        # Iterate through Linear + Activation pairs for hidden layers
        for i in range(self.num_hidden_layers):
            linear_layer = self.layers[i * 2]
            act_layer = self.layers[i * 2 + 1]
            current_activation = act_layer(linear_layer(current_activation))
        # Pass through final output layer
        logits = self.output_layer(current_activation)
        return logits

    def forward_with_intermediate_activations(
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """Forward pass that returns all intermediate activations.

        This returns the activations *after* the activation function for the
        input and each hidden layer. These are the 'a_i' vectors needed for
        local MF loss calculation and MF evaluation. Assumes input 'x' is

        Returns:
            A list of tensors `[a_0, a_1, ..., a_L]`, where `a_0` is the
            input and `a_i` (for i > 0) is the output of hidden layer
            (i-1)'s activation. The list length is num_hidden_layers + 1.
        """
        layer_activations = [x]  # a_0 is the input
        current_activation = x
        # Iterate through Linear + Activation pairs for hidden layers
        for i in range(self.num_hidden_layers):
            linear_layer = self.layers[i * 2]  # W_{i+1}
            act_layer = self.layers[i * 2 + 1]  # sigma_{i+1}
            pre_activation = linear_layer(current_activation)  # z_{i+1}
            current_activation = act_layer(pre_activation)  # This is a_{i+1}
            layer_activations.append(current_activation)

        # Note: a_L is the output of the L-th hidden layer's activation,
        # located at index L in the returned list.
        if len(layer_activations) != self.num_hidden_layers + 1:
            logger.warning(
                "Activation list length mismatch. Expected %d, got %d",
                self.num_hidden_layers + 1,
                len(layer_activations),
            )

        return layer_activations
