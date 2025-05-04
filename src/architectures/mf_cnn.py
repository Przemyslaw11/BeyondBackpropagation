# --------------------------------------------------------------------------------
# File: ./src/architectures/mf_cnn.py (NEW FILE)
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Type
import math
import logging

logger = logging.getLogger(__name__)

class MF_CNN_Block(nn.Module):
    """A single CNN block for the MF_CNN: Conv2d -> ReLU -> AvgPool2d"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Conv2d: 3x3 kernel, padding=1 to preserve spatial dims before pooling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        # AvgPool2d: 2x2 kernel, stride=2 to halve spatial dimensions
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Initialize weights (Kaiming for ReLU)
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5), nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class MF_CNN(nn.Module):
    """
    Convolutional Neural Network designed for the Mono-Forward (MF) algorithm,
    based on the 4-layer architecture described in Gong et al., 2025.
    Includes CNN blocks (W_i) and learnable projection matrices (M_i).
    """
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        image_size: int, # Needed to calculate flattened dimensions
        activation: str = "relu", # Keep for consistency, though block uses ReLU
        bias: bool = True, # Bias default for Conv layers
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_size = image_size

        if activation.lower() != "relu":
            logger.warning(f"MF_CNN explicitly uses ReLU in blocks, ignoring activation='{activation}'.")

        # --- CNN Blocks (W_1 to W_4) ---
        block_channels = [64, 128, 256, 512]
        self.blocks = nn.ModuleList()
        current_channels = input_channels
        for i, out_channels in enumerate(block_channels):
            block = MF_CNN_Block(current_channels, out_channels)
            self.blocks.append(block)
            current_channels = out_channels
        self.num_cnn_layers = len(self.blocks) # Should be 4

        # --- Calculate Flattened Dimensions & Create Projection Matrices (M_i) ---
        # M_i projects the output activation a_i (after pooling)
        # Need M_0 (for input a_0) to M_4 (for output a_4 of last block)
        self.projection_matrices = nn.ParameterList()
        self._activation_dims_flat: List[int] = [] # Store flattened dims of a_0, a_1, ..., a_4

        # Calculate dimensions step-by-step
        current_h, current_w = image_size, image_size
        # Dim for a_0 (input)
        a0_flat_dim = input_channels * current_h * current_w
        self._activation_dims_flat.append(a0_flat_dim)
        m0 = nn.Parameter(torch.empty(num_classes, a0_flat_dim))
        nn.init.kaiming_uniform_(m0, a=math.sqrt(5))
        self.projection_matrices.append(m0)

        # Dims for a_1 to a_4 (outputs of blocks 0 to 3)
        temp_channels = input_channels
        with torch.no_grad(): # Use dummy forward to get shapes reliably
            dummy_tensor = torch.zeros(1, input_channels, current_h, current_w)
            for i, block in enumerate(self.blocks):
                dummy_tensor = block(dummy_tensor)
                _, c_out, h_out, w_out = dummy_tensor.shape
                flat_dim = c_out * h_out * w_out
                self._activation_dims_flat.append(flat_dim)

                # Create M_{i+1}
                m_matrix = nn.Parameter(torch.empty(num_classes, flat_dim))
                nn.init.kaiming_uniform_(m_matrix, a=math.sqrt(5))
                self.projection_matrices.append(m_matrix)
                temp_channels = c_out # Not really needed here

        # Final classifier layer (used ONLY for BP baseline evaluation)
        # Input dim is the flattened output dim of the last block (a_4)
        self.final_flat_dim = self._activation_dims_flat[-1]
        self.output_layer = nn.Linear(self.final_flat_dim, num_classes, bias=bias)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        if bias and self.output_layer.bias is not None: nn.init.zeros_(self.output_layer.bias)

        logger.info(f"Initialized MF_CNN with {self.num_cnn_layers} blocks.")
        logger.info(f"Block output channels: {input_channels} -> {' -> '.join(map(str, block_channels))}")
        logger.info(f"Activation flattened dims (a_0 to a_{self.num_cnn_layers}): {self._activation_dims_flat}")
        logger.info(f"Created {len(self.projection_matrices)} projection matrices (M_0 to M_{self.num_cnn_layers}).")
        logger.info(f"Final flattened dim for BP baseline classifier: {self.final_flat_dim}")


    def get_projection_matrix(self, m_index: int) -> nn.Parameter:
        """Safely retrieves a projection matrix parameter by its index (0 to num_cnn_layers)."""
        if 0 <= m_index < len(self.projection_matrices):
            return self.projection_matrices[m_index]
        else:
            raise IndexError(f"Projection matrix index {m_index} out of bounds for {len(self.projection_matrices)} matrices.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the CNN blocks, returning final logits.
        Used ONLY for the BP baseline evaluation.
        """
        current_activation = x
        for block in self.blocks:
            current_activation = block(current_activation)
        # Flatten before final layer
        flat_features = current_activation.view(current_activation.size(0), -1)
        logits = self.output_layer(flat_features)
        return logits

    def forward_with_intermediate_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning the input and the activations *after pooling*
        for each block, flattened.
        These are the 'a_i' vectors needed for local MF loss calculation.

        Returns:
            List[torch.Tensor]: layer_activations = [a_0_flat, a_1_flat, ..., a_L_flat] (L=num_cnn_layers)
        """
        # Flatten input a_0
        a_0_flat = x.view(x.size(0), -1)
        layer_activations_flat = [a_0_flat]

        current_activation = x
        for block in self.blocks:
            current_activation = block(current_activation)
            # Flatten the output of the block (after pooling)
            a_i_flat = current_activation.view(current_activation.size(0), -1)
            layer_activations_flat.append(a_i_flat)

        if len(layer_activations_flat) != self.num_cnn_layers + 1:
            logger.warning(f"CNN activation list length mismatch. Expected {self.num_cnn_layers + 1}, got {len(layer_activations_flat)}")

        return layer_activations_flat

    def get_block_params(self, block_index: int) -> List[nn.Parameter]:
        """Returns parameters of a specific CNN block."""
        if 0 <= block_index < self.num_cnn_layers:
            return list(self.blocks[block_index].parameters())
        else:
            raise IndexError(f"Block index {block_index} out of range.")

    def get_intermediate_activation_shapes(self) -> List[Tuple[int, ...]]:
        """Helper to get theoretical shapes after each block (debugging)."""
        shapes = []
        c, h, w = self.input_channels, self.image_size, self.image_size
        shapes.append((c, h, w)) # Input shape
        dummy_tensor = torch.zeros(1, c, h, w)
        with torch.no_grad():
            for block in self.blocks:
                 dummy_tensor = block(dummy_tensor)
                 shapes.append(tuple(dummy_tensor.shape[1:])) # C, H, W
        return shapes