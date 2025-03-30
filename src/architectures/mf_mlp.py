# File: src/architectures/mf_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Type  # Added Type
import math  # For Kaiming init
import logging

logger = logging.getLogger(__name__)


class MF_MLP(nn.Module):
    """
    Multi-Layer Perceptron designed for the Mono-Forward (MF) algorithm.
    Includes standard feedforward layers (W_i) and learnable projection matrices (M_i).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = "relu",
        bias: bool = True,
    ):
        """
        Initializes the MF_MLP.

        Args:
            input_dim: Dimensionality of the input features (e.g., flattened image).
            hidden_dims: List of integers specifying the size of each hidden layer.
            num_classes: Number of output classes.
            activation: Activation function ('relu', 'tanh', etc.).
            bias: Whether to include bias terms in linear layers.
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
        # Add other activations if needed
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # --- Standard Feedforward Layers (W_i) ---
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            linear_layer = nn.Linear(current_dim, h_dim, bias=bias)
            # Apply initialization (e.g., Kaiming for ReLU)
            # nn.init.kaiming_uniform_(linear_layer.weight, a=math.sqrt(5), nonlinearity=activation.lower())
            # if bias: nn.init.zeros_(linear_layer.bias)
            self.layers.append(linear_layer)  # Linear layer for W_i
            self.layers.append(act_cls())  # Activation function
            current_dim = h_dim
        # Final classifier layer (W_L) - crucial for BP-style prediction and BP baseline
        self.output_layer = nn.Linear(current_dim, num_classes, bias=bias)

        # --- Learnable Projection Matrices (M_i) ---
        # M_i projects the activation of layer i (a_i) to goodness scores.
        # Dimensions: num_classes x num_neurons_in_layer_i
        # We need one M matrix for each hidden layer activation (a_1 to a_{L-1})
        self.projection_matrices = nn.ParameterList()
        for i, layer_dim in enumerate(hidden_dims):  # layer_dim = size of a_i
            # M_i corresponds to activation a_i (output of hidden layer i)
            m_matrix = nn.Parameter(torch.empty(num_classes, layer_dim))
            nn.init.kaiming_uniform_(
                m_matrix, a=math.sqrt(5)
            )  # Initialize M_i (or other scheme)
            self.projection_matrices.append(m_matrix)
            # logger.debug(f"Created projection matrix M_{i} with shape {m_matrix.shape}")

        logger.info(f"Initialized MF_MLP with {len(hidden_dims)} hidden layers.")
        layer_dims_str = " -> ".join(
            map(str, [input_dim] + hidden_dims + [num_classes])
        )
        logger.info(f"Feedforward Layer dimensions (W): {layer_dims_str}")
        logger.info(
            f"Created {len(self.projection_matrices)} projection matrices (M_0 to M_{self.num_hidden_layers-1})"
        )

    def get_projection_matrix(self, layer_index: int) -> nn.Parameter:
        """Safely retrieves a projection matrix parameter by its hidden layer index (0 to num_hidden_layers - 1)."""
        if 0 <= layer_index < len(self.projection_matrices):
            return self.projection_matrices[layer_index]
        else:
            raise IndexError(
                f"Projection matrix index {layer_index} out of bounds for {len(self.projection_matrices)} matrices."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the MLP layers (W_0...W_L), returning final logits.
        Used for inference (BP-style) and for the BP baseline. Assumes input 'x' is already flattened.
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
        """
        Forward pass returning the activations *after* the activation function
        for each hidden layer. Activation 0 is the input x.
        These are the 'a_i' vectors needed for local MF loss calculation.
        Assumes input 'x' is already flattened.

        Returns:
            List[torch.Tensor]: layer_activations = [a_0, a_1, ..., a_{L-1}]
            where a_0 = x (input), and a_i is the output of hidden layer i-1's activation (for i>0).
            So, a_1 is output of layer 0, a_2 is output of layer 1, etc.
        """
        layer_activations = [x]  # a_0 is the input
        current_activation = x
        # Iterate through Linear + Activation pairs for hidden layers
        for i in range(self.num_hidden_layers):
            linear_layer = self.layers[i * 2]
            act_layer = self.layers[i * 2 + 1]
            current_activation = linear_layer(current_activation)
            current_activation = act_layer(current_activation)  # This is a_{i+1}
            layer_activations.append(current_activation)

        # We only need hidden activations a_1 to a_{num_hidden_layers} for M_0 to M_{num_hidden_layers-1}
        # The list returned has length num_hidden_layers + 1
        return layer_activations


# Removed the __main__ block for cleaner architecture file
