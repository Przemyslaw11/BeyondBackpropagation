import torch
import torch.nn as nn
from typing import List, Union, Optional

class FF_Layer(nn.Module):
    """
    A single layer for the Forward-Forward algorithm, including normalization.
    """
    def __init__(self, in_features: int, out_features: int, activation: nn.Module = nn.ReLU(), normalize: bool = True, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        self.normalize = normalize
        self.layer_norm = nn.LayerNorm(out_features) if normalize else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer. Normalization is applied *before* activation
        as suggested in some FF implementations to keep activations in a good range.
        """
        x = self.linear(x)
        if self.normalize:
            # Normalize across the feature dimension
            x = self.layer_norm(x)
        x = self.activation(x)
        return x

    def forward_with_goodness(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns the sum of squared activities (goodness).
        Goodness is typically calculated *after* activation.
        """
        x_out = self.forward(x)
        goodness = torch.sum(x_out**2, dim=1) # Sum squared activities across feature dimension
        return x_out, goodness


class FF_MLP(nn.Module):
    """
    Multi-Layer Perceptron specifically designed for the Forward-Forward algorithm.

    Handles label embedding and layer-wise processing.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, activation: str = 'relu', normalize_layers: bool = True, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.normalize_layers = normalize_layers

        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'tanh':
            act_fn = nn.Tanh()
        # Add other activations if needed
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Layer definitions
        self.layers = nn.ModuleList()
        current_dim = input_dim + num_classes # Input dim + embedded label dim

        for h_dim in hidden_dims:
            self.layers.append(FF_Layer(current_dim, h_dim, activation=act_fn, normalize=normalize_layers, bias=bias))
            current_dim = h_dim

        # Note: FF typically doesn't have a final output layer in the same way as BP.
        # Classification is often done by comparing goodness of the last hidden layer
        # for positive vs negative examples, or by training a final linear layer separately.
        # We will handle the training logic (including loss/goodness comparison)
        # in the algorithm file (src/algorithms/ff.py).

        print(f"Initialized FF_MLP with {len(self.layers)} hidden layers.")
        print(f"Layer dimensions (input including label embedding): {input_dim + num_classes} -> {' -> '.join(map(str, hidden_dims))}")


    def embed_labels(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Embeds labels into the input tensor using one-hot encoding.
        Assumes the first 'num_classes' dimensions of the input are reserved for labels.
        """
        # Ensure input tensor has space for labels (optional, could be done externally)
        # if x.shape[1] == self.input_dim:
        #     x_padded = torch.zeros((x.shape[0], self.input_dim + self.num_classes), device=x.device)
        #     x_padded[:, self.num_classes:] = x
        #     x = x_padded
        # elif x.shape[1] != self.input_dim + self.num_classes:
        #      raise ValueError(f"Input tensor shape {x.shape} incompatible with input_dim {self.input_dim} and num_classes {self.num_classes}")

        # Create one-hot labels
        one_hot_labels = nn.functional.one_hot(labels, num_classes=self.num_classes).float()

        # Embed labels into the first num_classes dimensions
        x_with_labels = x.clone() # Avoid modifying original input if it's reused
        x_with_labels[:, :self.num_classes] = one_hot_labels
        return x_with_labels

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None, embed_labels: bool = True) -> List[torch.Tensor]:
        """
        Standard forward pass, returning activations from all layers.
        If labels are provided and embed_labels is True, embeds them first.
        """
        if embed_labels and labels is not None:
             # Pad input if necessary before embedding
             if x.shape[1] == self.input_dim:
                 x_padded = torch.zeros((x.shape[0], self.input_dim + self.num_classes), device=x.device)
                 x_padded[:, self.num_classes:] = x
                 x = x_padded
             elif x.shape[1] != self.input_dim + self.num_classes:
                  raise ValueError(f"Input tensor shape {x.shape} incompatible with input_dim {self.input_dim} and num_classes {self.num_classes}")
             x = self.embed_labels(x, labels)
        elif x.shape[1] != self.input_dim + self.num_classes:
             # If not embedding labels, input should already have the combined dimension
             raise ValueError(f"Input tensor shape {x.shape} does not match expected combined dimension {self.input_dim + self.num_classes} when embed_labels=False")


        activations = [x] # Store initial input (with embedded labels if applicable)
        current_activation = x
        for layer in self.layers:
            current_activation = layer(current_activation)
            activations.append(current_activation)
        return activations

    def forward_goodness_per_layer(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass specifically for FF training.
        Calculates and returns the 'goodness' (sum of squared activities) for each layer's output.
        Assumes input x already has labels embedded.
        """
        if x.shape[1] != self.input_dim + self.num_classes:
             raise ValueError(f"Input tensor shape {x.shape} does not match expected combined dimension {self.input_dim + self.num_classes}")

        layer_goodness = []
        current_activation = x
        for layer in self.layers:
            # Use the layer's specific forward_with_goodness method
            current_activation, goodness = layer.forward_with_goodness(current_activation)
            layer_goodness.append(goodness)

        return layer_goodness


if __name__ == '__main__':
    print("\nTesting FF_MLP...")
    # Config based on F-MNIST plan: 4x2000 ReLU MLP
    input_dim_fmnist = 28 * 28 # 784
    num_classes_fmnist = 10
    hidden_dims_fmnist = [2000, 2000, 2000, 2000]
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = FF_MLP(
        input_dim=input_dim_fmnist,
        hidden_dims=hidden_dims_fmnist,
        num_classes=num_classes_fmnist,
        activation='relu',
        normalize_layers=True
    ).to(device)
    print(model)

    # Create dummy data
    # Input without label embedding space
    dummy_images = torch.randn(batch_size, input_dim_fmnist).to(device)
    # Input with label embedding space (zeros initially)
    dummy_input_padded = torch.zeros(batch_size, input_dim_fmnist + num_classes_fmnist).to(device)
    dummy_input_padded[:, num_classes_fmnist:] = dummy_images
    dummy_labels = torch.randint(0, num_classes_fmnist, (batch_size,)).to(device)

    # --- Test label embedding ---
    print("\nTesting label embedding...")
    embedded_input = model.embed_labels(dummy_input_padded, dummy_labels)
    print("Original padded input shape:", dummy_input_padded.shape)
    print("Embedded input shape:", embedded_input.shape)
    assert embedded_input.shape == dummy_input_padded.shape
    # Check if labels are correctly placed
    for i in range(batch_size):
        label = dummy_labels[i].item()
        assert embedded_input[i, label] == 1.0
        assert embedded_input[i, :num_classes_fmnist].sum() == 1.0 # Only one hot bit
        # Check if image data is preserved
        assert torch.allclose(embedded_input[i, num_classes_fmnist:], dummy_images[i])
    print("Label embedding test passed.")

    # --- Test standard forward pass (with embedding) ---
    print("\nTesting standard forward pass (with label embedding)...")
    activations = model.forward(dummy_images, labels=dummy_labels, embed_labels=True)
    print(f"Number of activation tensors returned: {len(activations)} (Input + {len(model.layers)} layers)")
    assert len(activations) == len(model.layers) + 1
    print("Activation shapes:")
    print(f"  Input (embedded): {activations[0].shape}")
    assert activations[0].shape == (batch_size, input_dim_fmnist + num_classes_fmnist)
    for i, act in enumerate(activations[1:]):
        print(f"  Layer {i+1}: {act.shape}")
        assert act.shape == (batch_size, model.hidden_dims[i])

    # --- Test forward pass (without embedding, pre-formatted input) ---
    print("\nTesting standard forward pass (pre-formatted input)...")
    activations_preformatted = model.forward(embedded_input, embed_labels=False)
    assert len(activations_preformatted) == len(activations)
    for i in range(len(activations)):
         assert torch.allclose(activations_preformatted[i], activations[i])
    print("Forward pass with pre-formatted input matches.")


    # --- Test goodness calculation ---
    print("\nTesting forward goodness calculation...")
    layer_goodness = model.forward_goodness_per_layer(embedded_input)
    print(f"Number of goodness tensors returned: {len(layer_goodness)}")
    assert len(layer_goodness) == len(model.layers)
    print("Goodness shapes:")
    for i, goodness in enumerate(layer_goodness):
        print(f"  Layer {i+1}: {goodness.shape}")
        # Goodness should be a scalar per batch item
        assert goodness.shape == (batch_size,)
        assert torch.all(goodness >= 0) # Goodness should be non-negative
    print("Goodness calculation test passed.")
