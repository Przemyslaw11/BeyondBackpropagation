# File: src/architectures/mf_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MF_MLP(nn.Module):
    """
    Multi-Layer Perceptron designed for the Mono-Forward (MF) algorithm.
    Includes standard feedforward layers (W_i) and *learnable* projection matrices (M_i).
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
            input_dim: Dimensionality of the input features.
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
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        # Add other activations if needed
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # --- Standard Feedforward Layers (W_i) ---
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            linear_layer = nn.Linear(current_dim, h_dim, bias=bias)
            self.layers.append(linear_layer)  # Linear layer for W_i
            self.layers.append(act_fn())  # Activation function
            current_dim = h_dim
        # Final classifier layer (can be trained by BP or potentially MF style)
        self.output_layer = nn.Linear(current_dim, num_classes, bias=bias)

        # --- Learnable Projection Matrices (M_i) ---
        # M_i projects the activation of layer i (a_i) to goodness scores.
        # Dimensions: num_classes x num_neurons_in_layer_i
        self.projection_matrices = nn.ParameterList()
        all_layer_dims = hidden_dims  # Output dimensions of hidden layers 0 to L-1

        for i, layer_dim in enumerate(all_layer_dims):
            # M_i corresponds to activation a_i (output of hidden layer i)
            m_matrix = nn.Parameter(torch.empty(num_classes, layer_dim))
            nn.init.kaiming_uniform_(m_matrix, a=math.sqrt(5))  # Initialize M_i
            self.projection_matrices.append(m_matrix)

        print(f"Initialized MF_MLP with {len(hidden_dims)} hidden layers.")
        print(
            f"Layer dimensions (W): {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}"
        )
        print(
            f"Created {len(self.projection_matrices)} learnable projection matrices (M_0 to M_{self.num_hidden_layers-1})"
        )
        for i, M in enumerate(self.projection_matrices):
            print(f"  M_{i} shape: {M.shape}")  # Shape [num_classes, hidden_dims[i]]

    def get_projection_matrix(self, layer_index: int) -> nn.Parameter:
        """Safely retrieves a projection matrix parameter by its index (0 to num_hidden_layers - 1)."""
        if 0 <= layer_index < len(self.projection_matrices):
            return self.projection_matrices[layer_index]
        else:
            raise IndexError(
                f"Projection matrix index {layer_index} out of bounds for {len(self.projection_matrices)} matrices."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the MLP layers, returning final logits.
        Used for inference (BP-style) and potentially for the BP baseline.
        """
        # Flatten input if needed (assuming adapter handled it before calling)
        # x = x.view(x.shape[0], -1) # Ensure flattened
        current_activation = x
        # Iterate through Linear + Activation pairs for hidden layers
        for i in range(0, len(self.layers), 2):
            linear_layer = self.layers[i]
            act_layer = self.layers[i + 1]
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
        These are the 'a_i' vectors needed for local loss calculation.

        Returns:
            List[torch.Tensor]: layer_activations = [a_0, a_1, ..., a_{L-1}]
            where a_0 = x (input), and a_i is the output of hidden layer i.
        """
        # x = x.view(x.shape[0], -1) # Ensure flattened input
        layer_activations = [x]  # a_0 is the input
        current_activation = x
        # Iterate through Linear + Activation pairs for hidden layers
        for i in range(0, len(self.layers), 2):  # Step by 2 (Linear, Activation)
            linear_layer = self.layers[i]
            act_layer = self.layers[i + 1]
            current_activation = linear_layer(current_activation)
            current_activation = act_layer(
                current_activation
            )  # This is a_i (i=layer_index+1)
            layer_activations.append(current_activation)

        # We don't necessarily need the final logits here, just the hidden activations
        return layer_activations


# --- Keep the __main__ block for testing, adapting it as needed ---
import math

if __name__ == "__main__":
    print("\nTesting MF_MLP (Corrected)...")
    # Config based on F-MNIST plan: 2x1000 ReLU MLP
    input_dim_fmnist = 28 * 28  # 784
    num_classes_fmnist = 10
    hidden_dims_fmnist = [1000, 1000]
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create F-MNIST model
    print("\n--- F-MNIST Config (2x1000) ---")
    model_fmnist = MF_MLP(
        input_dim=input_dim_fmnist,
        hidden_dims=hidden_dims_fmnist,
        num_classes=num_classes_fmnist,
        activation="relu",
    ).to(device)
    # print(model_fmnist) # Print model structure if needed

    # Check projection matrices (should be Parameters)
    print("Projection Matrix Check:")
    num_expected_matrices = len(hidden_dims_fmnist)
    assert len(model_fmnist.projection_matrices) == num_expected_matrices
    for i in range(num_expected_matrices):
        m = model_fmnist.get_projection_matrix(i)
        print(f"  M_{i}: shape={m.shape}, requires_grad={m.requires_grad}")
        assert m.shape == (num_classes_fmnist, hidden_dims_fmnist[i])
        assert isinstance(m, nn.Parameter)
        assert m.requires_grad  # Crucial: should be True

    # Create dummy data (flattened)
    dummy_images_flat = torch.randn(batch_size, input_dim_fmnist).to(device)

    # --- Test standard forward pass ---
    print("\nTesting standard forward pass...")
    logits = model_fmnist.forward(dummy_images_flat)
    print("Logits shape:", logits.shape)
    assert logits.shape == (batch_size, num_classes_fmnist)

    # --- Test forward pass with intermediate activations ---
    print("\nTesting forward pass with intermediate activations...")
    activations_inter = model_fmnist.forward_with_intermediate_activations(
        dummy_images_flat
    )
    print(f"Number of activation tensors: {len(activations_inter)}")
    # Expect input (a_0) + one activation per hidden layer (a_1, a_2)
    assert len(activations_inter) == model_fmnist.num_hidden_layers + 1
    print("Activation shapes:")
    print(f"  a_0 (Input): {activations_inter[0].shape}")
    assert activations_inter[0].shape == (batch_size, input_dim_fmnist)
    for i, act in enumerate(activations_inter[1:]):  # a_1, a_2, ...
        print(f"  a_{i+1} (Layer {i} output): {act.shape}")
        assert act.shape == (batch_size, hidden_dims_fmnist[i])

    print("\nMF_MLP (Corrected) tests passed.")
