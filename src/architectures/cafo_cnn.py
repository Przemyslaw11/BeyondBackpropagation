# File: src/architectures/cafo_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Type  # Added Type
import logging

logger = logging.getLogger(__name__)


class CaFoBlock(nn.Module):
    """
    A single block for the CaFo CNN, typically: Conv -> BN -> Activation -> Pool.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        activation_cls: Type[nn.Module] = nn.ReLU,  # Pass class
        use_batchnorm: bool = True,
    ):
        super().__init__()
        # Use bias=False in Conv2d if BatchNorm is used, as BN has its own bias (beta)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = activation_cls()  # Instantiate activation
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self._output_shape = None  # Cache output shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pool(x)
        # Cache shape after first forward pass if needed (useful for predictor dim calculation)
        # if self._output_shape is None and not self.training: # Maybe cache only during eval?
        #     self._output_shape = x.shape
        return x

    def get_output_shape(
        self, input_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Calculates the output shape (C, H, W) for a given input shape (C, H, W)"""
        with torch.no_grad():
            # Move dummy tensor to the same device as the layer's parameters if possible
            device = (
                next(self.parameters()).device
                if list(self.parameters())
                else torch.device("cpu")
            )
            dummy_input = torch.zeros(1, *input_shape, device=device)
            # Ensure block is also on the same device
            self.to(device)
            dummy_output = self.forward(dummy_input)
            return dummy_output.shape[1:]  # Return C, H, W


class CaFoPredictor(nn.Module):
    """
    A simple predictor head (typically Linear) attached to the output of a CaFoBlock.
    """

    def __init__(self, in_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)
        logger.debug(f"CaFoPredictor: In={in_features}, Out={num_classes}, Bias={bias}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is expected to be the feature map output of a CaFoBlock [B, C, H, W]
        x = self.flatten(x)  # Flatten to [B, C*H*W]
        return self.fc(x)


class CaFo_CNN(nn.Module):
    """
    Cascaded Forward (CaFo) Convolutional Neural Network.
    Comprises multiple CaFoBlocks. Predictors are often handled externally during training.
    """

    def __init__(
        self,
        input_channels: int,
        block_channels: List[int],
        image_size: int,  # Needed to calculate dimensions
        num_classes: int,  # Needed for predictor instantiation if done here
        activation: str = "relu",
        use_batchnorm: bool = True,
        # use_predictors: bool = True # Flag removed, predictors handled externally
    ):
        super().__init__()
        self.input_channels = input_channels
        self.block_channels = block_channels
        self.image_size = image_size
        self.num_classes = num_classes  # Store for potential external use

        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.blocks = nn.ModuleList()
        # Keep track of output shapes to calculate predictor input dims
        self._block_output_shapes: List[Tuple[int, int, int]] = []

        current_channels = input_channels
        current_feature_map_size = image_size
        current_input_shape = (
            current_channels,
            current_feature_map_size,
            current_feature_map_size,
        )

        for i, out_channels in enumerate(block_channels):
            block = CaFoBlock(
                current_channels,
                out_channels,
                activation_cls=act_cls,
                use_batchnorm=use_batchnorm,
            )
            self.blocks.append(block)

            # Calculate and store the output shape of this block
            try:
                output_shape = block.get_output_shape(current_input_shape)
                self._block_output_shapes.append(output_shape)
                current_input_shape = output_shape  # Use for next block's calculation
                current_channels = output_shape[0]  # Update channels
            except Exception as e:
                logger.error(
                    f"Failed to determine output shape for block {i} with input shape {current_input_shape}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Shape calculation failed for block {i}.") from e

        # Predictors are NOT part of the main model structure anymore for better separation
        # They will be created and managed by the CaFo training logic externally.
        # self.predictors = nn.ModuleList()
        # for i, output_shape in enumerate(self._block_output_shapes):
        #     in_features = output_shape[0] * output_shape[1] * output_shape[2]
        #     predictor = CaFoPredictor(in_features, num_classes)
        #     self.predictors.append(predictor)

        logger.info(f"Initialized CaFo_CNN base with {len(self.blocks)} blocks.")
        logger.info(
            f"Block channels: {input_channels} -> {' -> '.join(map(str, block_channels))}"
        )
        logger.info(f"Block output shapes (C, H, W): {self._block_output_shapes}")

    def get_predictor_input_dim(self, block_index: int) -> int:
        """Returns the expected flattened input dimension for the predictor of a given block."""
        if 0 <= block_index < len(self._block_output_shapes):
            shape = self._block_output_shapes[block_index]
            return shape[0] * shape[1] * shape[2]  # C * H * W
        else:
            raise IndexError(f"Block index {block_index} out of range.")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through blocks, returning intermediate block outputs.
        Predictor application is handled externally.

        Args:
            x: Input tensor (batch of images).

        Returns:
            block_outputs: List of output tensors from each CaFoBlock.
        """
        block_outputs = []
        current_activation = x

        for block in self.blocks:
            current_activation = block(current_activation)
            block_outputs.append(current_activation)

        return block_outputs

    def forward_blocks_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all convolutional blocks, returning only the final block's output.
        Useful for constructing a BP baseline externally.
        """
        current_activation = x
        for block in self.blocks:
            current_activation = block(current_activation)
        return current_activation


# Removed the __main__ block for cleaner architecture file
