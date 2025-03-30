# File: src/architectures/ff_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F  # Import F
from typing import List, Union, Optional, Tuple  # Add Tuple
import math  # For kaiming_uniform_ init if used elsewhere, not directly here


class FF_Layer(nn.Module):
    """
    A single layer for the Forward-Forward algorithm, including normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.ReLU(),
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        # --- Use LayerNorm for simplicity, Hinton's exact normalization can be complex ---
        # Consider replacing with simple L2 norm if layer_norm proves problematic:
        # self.norm_layer = lambda x: F.normalize(x, p=2, dim=1)
        self.norm_layer = (
            nn.LayerNorm(out_features, elementwise_affine=True)
            if normalize
            else nn.Identity()
        )  # Standard LayerNorm
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer. Applies Linear -> Norm -> Activation.
        """
        x = self.linear(x)
        # --- Apply normalization BEFORE activation ---
        # This is a common choice in FF to keep activations well-behaved for goodness calc.
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
        # Goodness: sum of squares of activations
        goodness = torch.sum(x_out.pow(2), dim=1)
        return x_out, goodness


class FF_MLP(nn.Module):
    """
    Multi-Layer Perceptron specifically designed for the Forward-Forward algorithm.
    Handles label embedding and layer-wise processing.
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
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.normalize_layers = normalize_layers

        if activation.lower() == "relu":
            act_fn = nn.ReLU()
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.layers = nn.ModuleList()
        # Input dimension includes space for embedded labels
        current_dim = input_dim + num_classes

        for h_dim in hidden_dims:
            self.layers.append(
                FF_Layer(
                    current_dim,
                    h_dim,
                    activation=act_fn(),
                    normalize=normalize_layers,
                    bias=bias,
                )
            )  # Pass activation instance
            current_dim = h_dim

        print(f"Initialized FF_MLP with {len(self.layers)} hidden layers.")
        print(
            f"Layer dimensions (input includes label embedding): {input_dim + num_classes} -> {' -> '.join(map(str, hidden_dims))}"
        )

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
            raise ValueError(
                f"Input image tensor has dimension {x_images.shape[1]}, expected {self.input_dim}"
            )

        # Create padded tensor
        x_padded = torch.zeros(
            (batch_size, self.input_dim + self.num_classes), device=device
        )
        x_padded[:, self.num_classes :] = x_images  # Place images after label space

        # Create one-hot labels
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()

        # Embed labels
        x_padded[:, : self.num_classes] = one_hot_labels
        return x_padded

    def forward_upto(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Performs a forward pass through layers 0 to layer_idx (exclusive).
        Assumes input 'x' already has labels embedded if necessary.

        Args:
            x: Input tensor (shape: [batch_size, input_dim + num_classes]).
            layer_idx: Index of the layer *not* to run (runs up to layer_idx - 1).

        Returns:
            Output activation tensor from layer layer_idx - 1.
        """
        if x.shape[1] != self.input_dim + self.num_classes:
            raise ValueError(
                f"Input tensor shape {x.shape} does not match expected combined dimension {self.input_dim + self.num_classes}"
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
        Forward pass specifically for FF evaluation/training.
        Calculates and returns the 'goodness' for each layer's output.
        Assumes input x already has labels embedded.
        """
        if x.shape[1] != self.input_dim + self.num_classes:
            raise ValueError(
                f"Input tensor shape {x.shape} does not match expected combined dimension {self.input_dim + self.num_classes}"
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


# Keep the __main__ block for testing, potentially add test for forward_upto
if __name__ == "__main__":
    print("Testing FF_MLP...")
    # Config based on F-MNIST plan: 4x2000 ReLU MLP
    input_dim_fmnist = 28 * 28  # 784
    num_classes_fmnist = 10
    # hidden_dims_fmnist = [2000, 2000, 2000, 2000] # Original
    hidden_dims_fmnist = [200, 150, 100]  # Smaller for testing __main__
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = FF_MLP(
        input_dim=input_dim_fmnist,
        hidden_dims=hidden_dims_fmnist,
        num_classes=num_classes_fmnist,
        activation="relu",
        normalize_layers=True,
    ).to(device)
    print(model)

    # Create dummy data
    dummy_images = torch.randn(batch_size, input_dim_fmnist).to(device)
    dummy_labels = torch.randint(0, num_classes_fmnist, (batch_size,)).to(device)

    # --- Test prepare_ff_input ---
    print("\nTesting prepare_ff_input...")
    ff_input = model.prepare_ff_input(dummy_images, dummy_labels)
    print("Original image shape:", dummy_images.shape)
    print("FF input shape:", ff_input.shape)
    assert ff_input.shape == (batch_size, input_dim_fmnist + num_classes_fmnist)
    # Check if labels are correctly placed
    for i in range(batch_size):
        label = dummy_labels[i].item()
        assert ff_input[i, label] == 1.0
        assert ff_input[i, :num_classes_fmnist].sum() == 1.0  # Only one hot bit
        # Check if image data is preserved
        assert torch.allclose(ff_input[i, num_classes_fmnist:], dummy_images[i])
    print("prepare_ff_input test passed.")

    # --- Test forward_upto ---
    print("\nTesting FF_MLP forward_upto...")
    out_l0 = model.forward_upto(ff_input, 0)  # Should be identical to input
    assert torch.allclose(out_l0, ff_input)
    print("forward_upto(0) OK")

    out_l1 = model.forward_upto(ff_input, 1)  # Output of layer 0
    assert out_l1.shape == (batch_size, hidden_dims_fmnist[0])
    print(f"forward_upto(1) shape: {out_l1.shape} OK")

    out_l2 = model.forward_upto(ff_input, 2)  # Output of layer 1
    assert out_l2.shape == (batch_size, hidden_dims_fmnist[1])
    print(f"forward_upto(2) shape: {out_l2.shape} OK")

    out_l3 = model.forward_upto(ff_input, 3)  # Output of layer 2 (last layer)
    assert out_l3.shape == (batch_size, hidden_dims_fmnist[2])
    print(f"forward_upto(3) shape: {out_l3.shape} OK")

    try:
        model.forward_upto(ff_input, 4)  # Out of bounds
    except ValueError:
        print("forward_upto(4) correctly raised ValueError OK")

    # --- Test goodness calculation ---
    print("\nTesting forward goodness calculation...")
    layer_goodness = model.forward_goodness_per_layer(ff_input)
    print(f"Number of goodness tensors returned: {len(layer_goodness)}")
    assert len(layer_goodness) == len(model.layers)
    print("Goodness shapes:")
    for i, goodness in enumerate(layer_goodness):
        print(f"  Layer {i+1}: {goodness.shape}")
        assert goodness.shape == (batch_size,)
        assert torch.all(goodness >= 0)
    print("Goodness calculation test passed.")
