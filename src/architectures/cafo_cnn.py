import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Type
import logging

logger = logging.getLogger(__name__)


class CaFoBlock(nn.Module):
    """
    A single block for the CaFo CNN, typically: Conv -> Activation -> Pool -> BN.
    MODIFIED: Changed layer order to Conv->Act->Pool->BN to match paper/reference.
    MODIFIED: Added explicit Kaiming uniform initialization for Conv layer.
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
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )

        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        logger.debug(f"CaFoBlock Conv ({in_channels}->{out_channels}): Explicitly applied Kaiming Uniform init.")

        self.activation = activation_cls()
        self.pool = (
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            if pool_kernel_size > 0
            else nn.Identity()
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.bn(x)
        return x

    def get_output_shape(
        self, input_shape: Tuple[int, int, int], device: Optional[torch.device] = None
    ) -> Tuple[int, int, int]:
        """Calculates the output shape (C, H, W) for a given input shape (C, H, W)"""
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
    """
    A simple predictor head (typically Linear) attached to the output of a CaFoBlock.
    """

    def __init__(self, in_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)
        logger.debug(f"CaFoPredictor: In={in_features}, Out={num_classes}, Bias={bias}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc(x)


class CaFo_CNN(nn.Module):
    """
    Cascaded Forward (CaFo) Convolutional Neural Network base.
    Contains only the cascaded blocks. Predictors are handled externally.
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
    ):
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
        except: device = device_to_check
        del temp_param

        for i, out_channels in enumerate(block_channels):
            # Determine padding based on kernel size to preserve dimensions if stride=1
            padding = (
                kernel_size // 2 if kernel_size % 2 != 0 else 0
            )  # Common padding for odd kernels

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
                    f"Failed to determine output shape for block {i} with input shape {current_input_shape}: {e}",
                    exc_info=True,
                )
                raise RuntimeError(f"Shape calculation failed for block {i}.") from e

        logger.info(f"Initialized CaFo_CNN base with {len(self.blocks)} blocks.")
        logger.info(
            f"Block channels: {input_channels} -> {' -> '.join(map(str, block_channels))}"
        )
        logger.info(f"Block output shapes (C, H, W): {self._block_output_shapes}")
        logger.info(f"Block output flattened dims: {self._block_output_dims_flat}")

    def get_predictor_input_dim(self, block_index: int) -> int:
        """Returns the expected flattened input dimension for the predictor of a given block."""
        if 0 <= block_index < len(self._block_output_dims_flat):
            return self._block_output_dims_flat[block_index]
        else:
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