import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class CaFoBlock(nn.Module):
    """
    A single block for the CaFo CNN, consisting of Conv -> BN -> ReLU -> MaxPool.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, pool_kernel_size: int = 2, pool_stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # BatchNorm handles the bias
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self._output_shape = None # Cache output shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        if self._output_shape is None:
             self._output_shape = x.shape # Cache shape after first forward
        return x

    def get_output_dim(self, input_shape: Tuple[int, int, int]) -> int:
        """ Calculates the flattened output dimension for a given input shape C, H, W """
        with torch.no_grad():
             dummy_input = torch.zeros(1, *input_shape)
             dummy_output = self.forward(dummy_input)
             return dummy_output.numel()


class CaFoPredictor(nn.Module):
    """
    A simple linear predictor head attached to the output of a CaFoBlock.
    """
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the input feature map from the block
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        elif x.dim() < 2:
             raise ValueError(f"Predictor input must have at least 2 dimensions (batch, features), got shape {x.shape}")
        return self.fc(x)


class CaFo_CNN(nn.Module):
    """
    Cascaded Forward (CaFo) Convolutional Neural Network.
    Comprises multiple CaFoBlocks, each followed by a local CaFoPredictor.
    Designed for layer-wise training as described in the CaFo paper.
    """
    def __init__(self, input_channels: int, block_channels: List[int], image_size: int, num_classes: int):
        super().__init__()
        self.input_channels = input_channels
        self.block_channels = block_channels
        self.image_size = image_size
        self.num_classes = num_classes

        self.blocks = nn.ModuleList()
        self.predictors = nn.ModuleList()

        current_channels = input_channels
        current_feature_map_size = image_size
        current_input_shape = (current_channels, current_feature_map_size, current_feature_map_size)

        for i, out_channels in enumerate(block_channels):
            block = CaFoBlock(current_channels, out_channels)
            self.blocks.append(block)

            # Calculate the flattened output dimension of this block
            # Use the helper method on the block
            try:
                 current_flattened_dim = block.get_output_dim(current_input_shape)
                 # Update shape for the next block's input calculation
                 # Need dummy pass to get H, W out
                 with torch.no_grad():
                      dummy_out = block(torch.zeros(1, *current_input_shape))
                      current_input_shape = dummy_out.shape[1:] # C, H, W
            except Exception as e:
                 print(f"Warning: Could not automatically determine output dimension for block {i}. Predictor input size might be incorrect. Error: {e}")
                 # Fallback or raise error - let's assume a large enough default? Risky.
                 # Or require manual calculation / configuration.
                 # For now, let's raise an error if it fails.
                 raise RuntimeError(f"Failed to determine output dimension for block {i} with input shape {current_input_shape}") from e


            predictor = CaFoPredictor(current_flattened_dim, num_classes)
            self.predictors.append(predictor)

            current_channels = out_channels # Input channels for the next block

        print(f"Initialized CaFo_CNN with {len(self.blocks)} blocks.")
        print(f"Block channels: {input_channels} -> {' -> '.join(map(str, block_channels))}")
        print(f"Predictors attached to each block output, predicting {num_classes} classes.")

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Full forward pass returning intermediate block outputs and all predictor outputs.
        Suitable for layer-wise CaFo training.

        Args:
            x: Input tensor (batch of images).

        Returns:
            A tuple containing:
            - block_outputs: List of output tensors from each CaFoBlock.
            - predictor_outputs: List of output tensors from each CaFoPredictor.
        """
        block_outputs = []
        predictor_outputs = []
        current_activation = x

        for i, block in enumerate(self.blocks):
            current_activation = block(current_activation)
            block_outputs.append(current_activation)

            predictor = self.predictors[i]
            pred_out = predictor(current_activation)
            predictor_outputs.append(pred_out)

        return block_outputs, predictor_outputs

    def forward_blocks_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional blocks only, returning the final block's output.
        Useful for analysis or potentially for constructing a BP baseline *externally*.
        """
        current_activation = x
        for block in self.blocks:
            current_activation = block(current_activation)
        return current_activation

    # Note: A forward_for_bp method using the last predictor is less ideal for baselines.
    # The BP baseline should ideally be constructed separately using CaFoBlocks
    # and a single final classifier added *after* the last block in the BP training script.


if __name__ == '__main__':
    print("\nTesting CaFo_CNN...")

    # Config based on plan: 3-Block CNN (32/128/512 channels)
    input_channels_cifar = 3
    num_classes_cifar = 10
    image_size_cifar = 32
    block_channels_cifar = [32, 128, 512]
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    try:
        model = CaFo_CNN(
            input_channels=input_channels_cifar,
            block_channels=block_channels_cifar,
            image_size=image_size_cifar,
            num_classes=num_classes_cifar
        ).to(device)
        print(model)

        # Create dummy data
        dummy_images = torch.randn(batch_size, input_channels_cifar, image_size_cifar, image_size_cifar).to(device)

        # --- Test full forward pass (for CaFo training) ---
        print("\nTesting full forward pass (blocks and predictors)...")
        block_outputs, predictor_outputs = model.forward(dummy_images)

        print(f"Number of block outputs: {len(block_outputs)}")
        assert len(block_outputs) == len(model.blocks)
        print("Block output shapes:")
        for i, out in enumerate(block_outputs):
            print(f"  Block {i+1}: {out.shape}")
            # Check channel dimension matches config
            assert out.shape[1] == model.block_channels[i]
            # Check spatial dimensions are decreasing (due to pooling)
            if i > 0:
                 assert out.shape[2] < block_outputs[i-1].shape[2]
                 assert out.shape[3] < block_outputs[i-1].shape[3]

        print(f"\nNumber of predictor outputs: {len(predictor_outputs)}")
        assert len(predictor_outputs) == len(model.predictors)
        print("Predictor output shapes:")
        for i, pred in enumerate(predictor_outputs):
            print(f"  Predictor {i+1}: {pred.shape}")
            assert pred.shape == (batch_size, num_classes_cifar)

        # --- Test forward pass (blocks only) ---
        print("\nTesting forward pass (blocks only)...")
        final_block_output = model.forward_blocks_only(dummy_images)
        print(f"Final block output shape: {final_block_output.shape}")
        assert final_block_output.shape == block_outputs[-1].shape # Should match last block output

        print("\nCaFo_CNN tests passed.")

    except Exception as e:
        print(f"\nError during CaFo_CNN testing: {e}", exc_info=True)
