"""Implements the CaFo_CNN model for the Cascaded Forward algorithm."""

import logging
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CaFoBlock(nn.Module):
    """A single block for the CaFo CNN: Conv -> Activation -> Pool -> BN.

    MODIFIED: Changed layer order to Conv->Act->Pool->BN to match the paper.
    MODIFIED: Added explicit Kaiming uniform initialization for the Conv layer.
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
        activation_cls: Type[nn.Module] = nn.ReLU,
        use_batchnorm: bool = True,
    ) -> None:
        """Initializes a CaFoBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for the convolutional layer.
            stride: Stride for the convolutional layer.
            padding: Padding for the convolutional layer.
            pool_kernel_size: Kernel size for the pooling layer.
            pool_stride: Stride for the pooling layer.
            activation_cls: The activation function class (e.g., nn.ReLU).
            use_batchnorm: Whether to use batch normalization.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )

        nn.init.kaiming_uniform_(self.conv.weight, mode="fan_in", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        logger.debug(
            "CaFoBlock Conv (%d->%d): Applied Kaiming Uniform init.",
            in_channels,
            out_channels,
        )

        self.activation = activation_cls()
        self.pool = (
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            if pool_kernel_size > 0
            else nn.Identity()
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the block."""
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.bn(x)
        return x

    def get_output_shape(
        self, input_shape: Tuple[int, int, int], device: Optional[torch.device] = None
    ) -> Tuple[int, int, int]:
        """Calculates the output shape (C, H, W) for a given input shape."""
        if device is None:
            device = (
                next(self.parameters()).device
                if list(self.parameters())
                else torch.device("cpu")
            )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape, device=device)
            self.to(device)
            dummy_output = self.forward(dummy_input)
            return dummy_output.shape[1:]  # Return C, H, W


class CaFoPredictor(nn.Module):
    """A simple linear predictor head attached to a CaFoBlock output."""

    def __init__(self, in_features: int, num_classes: int, bias: bool = True) -> None:
        """Initializes the CaFoPredictor.

        Args:
            in_features: Number of input features (flattened from block output).
            num_classes: Number of output classes.
            bias: Whether to use a bias term in the linear layer.
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)
        logger.debug(
            "CaFoPredictor: In=%d, Out=%d, Bias=%s", in_features, num_classes, bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass through the predictor."""
        x = self.flatten(x)
        return self.fc(x)


class CaFo_CNN(nn.Module):
    """Cascaded Forward (CaFo) Convolutional Neural Network base.

    This module contains only the cascaded blocks. Predictors are handled
    externally by the training logic.
    """

    def __init__(
        self,
        input_channels: int,
        block_channels: List[int],
        image_size: int,
        num_classes: int,
        activation: str = "relu",
        use_batchnorm: bool = True,
        kernel_size: int = 3,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
    ) -> None:
        """Initializes the CaFo_CNN base model.

        Args:
            input_channels: Number of channels in the input image.
            block_channels: A list of output channels for each CaFoBlock.
            image_size: The height and width of the input image.
            num_classes: The number of output classes.
            activation: The activation function to use ('relu' or 'tanh').
            use_batchnorm: Whether to use batch normalization in the blocks.
            kernel_size: Kernel size for the convolutional layers.
            pool_kernel_size: Kernel size for the pooling layers.
            pool_stride: Stride for the pooling layers.
        """
        super().__init__()
        self.input_channels = input_channels
        self.block_channels = block_channels
        self.image_size = image_size
        self.num_classes = num_classes

        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.blocks = nn.ModuleList()
        self._block_output_shapes: List[Tuple[int, int, int]] = []
        self._block_output_dims_flat: List[int] = []

        current_channels = input_channels
        current_feature_map_size = image_size
        current_input_shape = (
            current_channels,
            current_feature_map_size,
            current_feature_map_size,
        )

        device_to_check = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temp_param = nn.Parameter(torch.empty(0))
        try:
            device = temp_param.device
        except Exception:
            device = device_to_check
        del temp_param

        for i, out_channels in enumerate(block_channels):
            # Determine padding to preserve dimensions if stride=1
            padding = (
                kernel_size // 2 if kernel_size % 2 != 0 else 0
            )  # Common for odd kernels

            block = CaFoBlock(
                current_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                pool_kernel_size=pool_kernel_size,
                pool_stride=pool_stride,
                activation_cls=act_cls,
                use_batchnorm=use_batchnorm,
            )
            self.blocks.append(block)

            try:
                output_shape = block.get_output_shape(
                    current_input_shape, device=device
                )
                self._block_output_shapes.append(output_shape)
                flat_dim = output_shape[0] * output_shape[1] * output_shape[2]
                self._block_output_dims_flat.append(flat_dim)

                current_input_shape = output_shape
                current_channels = output_shape[0]
            except Exception as e:
                logger.error(
                    "Failed to get output shape for block %d with input shape %s: %s",
                    i,
                    current_input_shape,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(f"Shape calculation failed for block {i}.") from e

        logger.info("Initialized CaFo_CNN base with %d blocks.", len(self.blocks))
        logger.info(
            "Block channels: %d -> %s",
            input_channels,
            " -> ".join(map(str, block_channels)),
        )
        logger.info("Block output shapes (C, H, W): %s", self._block_output_shapes)
        logger.info("Block output flattened dims: %s", self._block_output_dims_flat)

    def get_predictor_input_dim(self, block_index: int) -> int:
        """Returns the flattened input dimension for a given block's predictor."""
        if 0 <= block_index < len(self._block_output_dims_flat):
            return self._block_output_dims_flat[block_index]
        raise IndexError(f"Block index {block_index} out of range.")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through blocks, returning intermediate block outputs."""
        block_outputs = []
        current_activation = x
        for block in self.blocks:
            current_activation = block(current_activation)
            block_outputs.append(current_activation)
        return block_outputs

    def forward_blocks_only(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the final block's output."""
        current_activation = x
        for block in self.blocks:
            current_activation = block(current_activation)
        return current_activation
