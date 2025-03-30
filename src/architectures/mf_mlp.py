import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class MF_MLP(nn.Module):
    """
    Multi-Layer Perceptron designed for the Mono-Forward (MF) algorithm.
    Includes standard feedforward layers and fixed random projection matrices (M_i).
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, activation: str = 'relu', bias: bool = True, projection_dim: Optional[int] = None):
        """
        Initializes the MF_MLP.

        Args:
            input_dim: Dimensionality of the input features.
            hidden_dims: List of integers specifying the size of each hidden layer.
            num_classes: Number of output classes.
            activation: Activation function ('relu', 'tanh', etc.).
            bias: Whether to include bias terms in linear layers.
            projection_dim: Dimensionality of the random projection space (d_p in the paper).
                            If None, defaults to num_classes.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.projection_dim = projection_dim if projection_dim is not None else num_classes

        if activation.lower() == 'relu':
            act_fn = nn.ReLU
        elif activation.lower() == 'tanh':
            act_fn = nn.Tanh
        # Add other activations if needed
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # --- Standard Feedforward Layers ---
        self.layers = nn.ModuleList()
        current_dim = input_dim
        all_dims = [input_dim] + hidden_dims
        for i, h_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(current_dim, h_dim, bias=bias))
            self.layers.append(act_fn())
            current_dim = h_dim
        # Final classifier layer
        self.output_layer = nn.Linear(current_dim, num_classes, bias=bias)

        # --- Fixed Random Projection Matrices (M_i) ---
        # These matrices project the *output* of each layer (including input)
        # into a fixed random subspace for the MF loss calculation.
        # They are NOT trainable parameters. We register them as buffers.
        self.projection_matrices = nn.ParameterList() # Use ParameterList to store tensors that should be moved to device etc. but not trained
        # Or use register_buffer if they strictly don't need gradients ever (safer)
        # Let's use register_buffer.

        proj_input_dims = [input_dim] + hidden_dims # Dimensions *before* projection
        for i, proj_in_dim in enumerate(proj_input_dims):
            # Create a fixed random matrix M_i: R^{d_p x d_i}
            # where d_i is the dimension of layer i's output (or input for i=0)
            # and d_p is the projection dimension.
            m_matrix = torch.randn(self.projection_dim, proj_in_dim)
            # Normalize the matrix (optional, but can help stability)
            m_matrix = F.normalize(m_matrix, p=2, dim=1)
            # Register as buffer (not trained, but part of the model state)
            self.register_buffer(f"projection_matrix_{i}", m_matrix)

        print(f"Initialized MF_MLP with {len(hidden_dims)} hidden layers.")
        print(f"Layer dimensions: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
        print(f"Registered {len(proj_input_dims)} projection matrices (M_0 to M_{len(hidden_dims)})")
        print(f"Projection dimension (d_p): {self.projection_dim}")


    def get_projection_matrix(self, layer_index: int) -> torch.Tensor:
        """ Safely retrieves a projection matrix by its index (0 to num_hidden_layers). """
        buffer_name = f"projection_matrix_{layer_index}"
        if hasattr(self, buffer_name):
             return getattr(self, buffer_name)
        else:
             raise IndexError(f"Projection matrix for layer index {layer_index} not found.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass through the MLP layers, returning final logits.
        This is used for inference and potentially for the BP baseline.
        """
        for layer in self.layers:
            x = layer(x)
        logits = self.output_layer(x)
        return logits

    def forward_with_intermediate_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that returns the final logits and the activations *before*
        the activation function for each layer (including the input).
        These pre-activations (or direct outputs of linear layers) might be needed
        for the MF loss calculation depending on the exact formulation.
        Alternatively, could return activations *after* activation function. Let's return *after*.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: (final_logits, layer_activations)
            layer_activations[0] is the input x.
            layer_activations[i] (i>0) is the output of the i-th hidden layer's activation.
        """
        layer_activations = [x] # Include input as activation 0
        current_activation = x
        # Iterate through Linear + Activation pairs
        for i in range(0, len(self.layers), 2): # Step by 2 (Linear, Activation)
             linear_layer = self.layers[i]
             act_layer = self.layers[i+1]
             current_activation = linear_layer(current_activation)
             current_activation = act_layer(current_activation)
             layer_activations.append(current_activation)

        logits = self.output_layer(current_activation)
        return logits, layer_activations


if __name__ == '__main__':
    print("\nTesting MF_MLP...")
    # Config based on F-MNIST plan: 2x1000 ReLU MLP
    input_dim_fmnist = 28 * 28 # 784
    num_classes_fmnist = 10
    hidden_dims_fmnist = [1000, 1000]
    projection_dim_fmnist = 10 # Example, could be num_classes or other value
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create F-MNIST model
    print("\n--- F-MNIST Config (2x1000) ---")
    model_fmnist = MF_MLP(
        input_dim=input_dim_fmnist,
        hidden_dims=hidden_dims_fmnist,
        num_classes=num_classes_fmnist,
        activation='relu',
        projection_dim=projection_dim_fmnist
    ).to(device)
    print(model_fmnist)

    # Check projection matrices
    print("Projection Matrix Shapes:")
    num_expected_matrices = len(hidden_dims_fmnist) + 1
    for i in range(num_expected_matrices):
        m = model_fmnist.get_projection_matrix(i)
        print(f"  M_{i}: {m.shape}")
        # Shape should be [projection_dim, layer_input_dim]
        layer_input_dim = input_dim_fmnist if i == 0 else hidden_dims_fmnist[i-1]
        assert m.shape == (projection_dim_fmnist, layer_input_dim)
        # Check if requires_grad is False (since it's a buffer)
        assert not m.requires_grad

    # Create dummy data
    dummy_images = torch.randn(batch_size, input_dim_fmnist).to(device)

    # --- Test standard forward pass ---
    print("\nTesting standard forward pass...")
    logits = model_fmnist.forward(dummy_images)
    print("Logits shape:", logits.shape)
    assert logits.shape == (batch_size, num_classes_fmnist)

    # --- Test forward pass with intermediate activations ---
    print("\nTesting forward pass with intermediate activations...")
    logits_inter, activations_inter = model_fmnist.forward_with_intermediate_activations(dummy_images)
    print("Logits shape (intermediate):", logits_inter.shape)
    assert torch.allclose(logits_inter, logits) # Logits should match standard forward
    print(f"Number of activation tensors: {len(activations_inter)}")
    assert len(activations_inter) == num_expected_matrices # Input + hidden layers
    print("Activation shapes:")
    print(f"  Input: {activations_inter[0].shape}")
    assert activations_inter[0].shape == (batch_size, input_dim_fmnist)
    for i, act in enumerate(activations_inter[1:]):
        print(f"  Layer {i+1}: {act.shape}")
        assert act.shape == (batch_size, hidden_dims_fmnist[i])

    # --- Test CIFAR Config ---
    print("\n--- CIFAR Config (3x2000) ---")
    input_dim_cifar = 32*32*3 # Flattened CIFAR image
    num_classes_cifar = 10 # or 100
    hidden_dims_cifar = [2000, 2000, 2000]
    projection_dim_cifar = num_classes_cifar

    model_cifar = MF_MLP(
        input_dim=input_dim_cifar,
        hidden_dims=hidden_dims_cifar,
        num_classes=num_classes_cifar,
        activation='relu',
        projection_dim=projection_dim_cifar
    ).to(device)
    print(model_cifar)

    # Quick check on CIFAR model forward pass
    dummy_cifar_flat = torch.randn(batch_size, input_dim_cifar).to(device)
    logits_cifar = model_cifar.forward(dummy_cifar_flat)
    assert logits_cifar.shape == (batch_size, num_classes_cifar)
    print(f"CIFAR model forward pass successful, logits shape: {logits_cifar.shape}")

    print("\nMF_MLP tests passed.")
