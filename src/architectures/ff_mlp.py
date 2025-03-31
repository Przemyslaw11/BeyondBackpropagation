# --------------------------------------------------------------------------------
# File: ./src/architectures/ff_mlp.py
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Type
import math  # Import math for initialization
import logging

logger = logging.getLogger(__name__)


class FF_Layer(nn.Module):
    """
    A single layer for the Forward-Forward algorithm, encapsulating the sequence:
    Linear -> Activation -> Normalization.

    This structure aligns with the concept of computing layer "goodness" based on
    activations *before* normalization removes magnitude information.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        activation_cls (Type[nn.Module]): The activation function class (e.g., nn.ReLU).
        normalize (bool): Whether to apply normalization after activation. Defaults to True.
        norm_type (str): Type of normalization ('length' or 'layernorm'). Defaults to 'length'.
        bias (bool): Whether the linear layer uses a bias term. Defaults to True.
        norm_eps (float): Epsilon added to the denominator for numerical stability
                          during normalization (especially length normalization).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_cls: Type[nn.Module] = nn.ReLU,
        normalize: bool = True,  # Default to true for FF
        norm_type: str = "length",  # 'length' or 'layernorm'
        bias: bool = True,
        norm_eps: float = 1e-8,  # Epsilon for normalization stability
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation_cls()
        self.norm_eps = norm_eps  # Store epsilon

        # --- Explicit Initialization ---
        # Apply standard initialization schemes
        if isinstance(self.activation, nn.ReLU):
            nn.init.kaiming_uniform_(
                self.linear.weight, a=math.sqrt(5), nonlinearity="relu"
            )
        elif isinstance(self.activation, nn.Tanh):
            # Xavier/Glorot might be better for tanh
            nn.init.xavier_uniform_(self.linear.weight)
        else:
            # Default Kaiming for other activations or custom ones
            nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

        if bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        # --- End Initialization ---

        self.norm_type = norm_type.lower()
        self.normalize = normalize
        if self.normalize:
            if self.norm_type == "layernorm":
                # Standard LayerNorm with learnable affine parameters
                self.norm_layer = nn.LayerNorm(
                    out_features, elementwise_affine=True, eps=norm_eps
                )
                logger.debug(
                    f"FF_Layer: Using nn.LayerNorm for normalization (eps={norm_eps})."
                )
            elif self.norm_type == "length":
                # Hinton's simplified length normalization (no learnable parameters)
                # Use manual implementation with epsilon for stability
                self.norm_layer = lambda x: x / (
                    torch.linalg.norm(x, dim=1, keepdim=True) + self.norm_eps
                )
                logger.debug(
                    f"FF_Layer: Using Length Normalization (L2 norm with eps={norm_eps})."
                )
            else:
                raise ValueError(
                    f"Unsupported norm_type: {norm_type}. Choose 'length' or 'layernorm'."
                )
        else:
            self.norm_layer = nn.Identity()

        logger.debug(
            f"FF_Layer: In={in_features}, Out={out_features}, Normalize={normalize} (Type: {norm_type if normalize else 'None'}), Bias={bias}, Activation={activation_cls.__name__}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Linear -> Activation -> Norm.
        Normalization is applied *after* activation. This is crucial for FF, as
        the 'goodness' is typically calculated based on the squared activations
        *before* the normalization step removes the magnitude information needed
        for the goodness signal (see Hinton, 2022, Sec 2.1). The normalized output
        is then passed to the next layer.
        """
        x_lin = self.linear(x)
        x_act = self.activation(x_lin)  # Activation applied first
        x_norm = self.norm_layer(x_act)  # Normalization applied last
        return x_norm

    def forward_with_goodness(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns the sum of squared activities (goodness).
        Goodness is calculated *after* activation but *before* final normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The normalized output activation tensor (passed to next layer).
                - The goodness score tensor (sum of squared activations before norm)
                  for each sample in the batch (shape: [batch_size]).
        """
        x_lin = self.linear(x)
        x_act = self.activation(x_lin)  # Calculate activation output
        # Goodness: sum of squares of activations (BEFORE final normalization)
        # The pow(2) and sum constitutes the goodness function G_l(x_l) for this layer.
        goodness = torch.sum(x_act.pow(2), dim=1)  # Sum over the feature dimension
        x_norm = self.norm_layer(x_act)  # Apply final normalization for output
        return x_norm, goodness


class FF_MLP(nn.Module):
    """
    Multi-Layer Perceptron specifically designed for the Forward-Forward algorithm.

    Structure:
    - Input Layer: A linear layer (`input_adapter_layer`) processes the flattened input
      (e.g., image with embedded labels), followed by activation and normalization.
      This acts as the first effective hidden layer (layer 0).
    - Hidden Layers: Subsequent hidden layers are instances of `FF_Layer`, each
      applying Linear -> Activation -> Normalization.
    - No final output layer for classification is typically part of the core FF_MLP
      during training, as inference involves multiple passes.

    Note: The handling of the first layer (`input_adapter_layer` + separate norm/activation)
    is slightly asymmetric compared to the subsequent `FF_Layer` instances. This design choice
    explicitly separates the initial input processing from the repeatable hidden layer block
    structure but achieves the same functional sequence (Linear -> Activation -> Norm)
    for the first effective hidden layer.

    Args:
        input_dim (int): Dimensionality of the flattened input (e.g., 784 for 28x28x1).
        hidden_dims (List[int]): List containing the number of neurons for each hidden layer.
                                 The first element is the output size of the input adapter layer.
        num_classes (int): Number of output classes (used for evaluation logic and determining
                           label embedding size if applicable).
        activation (str): Name of the activation function (e.g., 'relu', 'tanh'). Defaults to 'relu'.
        normalize_layers (bool): Whether to apply normalization in hidden layers. Defaults to True.
        norm_type (str): Type of normalization ('length' or 'layernorm'). Defaults to 'length'.
        bias (bool): Whether linear layers use bias terms. Defaults to True.
    """

    def __init__(
        self,
        input_dim: int,  # Should be image_size * image_size * channels
        hidden_dims: List[int],
        num_classes: int,
        activation: str = "relu",
        normalize_layers: bool = True,
        norm_type: str = "length",
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes  # Needed for evaluation logic
        self.normalize_layers = normalize_layers
        self.norm_type = norm_type

        if not hidden_dims:
            raise ValueError("hidden_dims list cannot be empty for FF_MLP.")

        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "tanh":
            act_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # --- Input Adapter Layer (Effective Hidden Layer 0) ---
        # Processes the flattened input (e.g., image with pixel-embedded labels)
        # In features = input_dim (e.g., 784), Out features = first hidden dim
        self.input_adapter_layer = nn.Linear(input_dim, hidden_dims[0], bias=bias)
        self.first_layer_activation = act_cls()

        # --- Explicit Initialization for input_adapter_layer ---
        if isinstance(self.first_layer_activation, nn.ReLU):
            nn.init.kaiming_uniform_(
                self.input_adapter_layer.weight, a=math.sqrt(5), nonlinearity="relu"
            )
        elif isinstance(self.first_layer_activation, nn.Tanh):
            nn.init.xavier_uniform_(self.input_adapter_layer.weight)
        else:
            nn.init.kaiming_uniform_(self.input_adapter_layer.weight, a=math.sqrt(5))
        if bias and self.input_adapter_layer.bias is not None:
            nn.init.zeros_(self.input_adapter_layer.bias)
        # --- End Initialization ---

        # Apply normalization to the output of the first effective hidden layer
        norm_eps = 1e-8  # Define epsilon for first layer norm as well
        if normalize_layers:
            if norm_type.lower() == "layernorm":
                self.first_layer_norm = nn.LayerNorm(hidden_dims[0], eps=norm_eps)
            elif norm_type.lower() == "length":
                self.first_layer_norm = lambda x: x / (
                    torch.linalg.norm(x, dim=1, keepdim=True) + norm_eps
                )
            else:  # Fallback to Identity if norm_type is invalid (already checked in FF_Layer)
                logger.warning(
                    f"Invalid norm_type '{norm_type}', disabling norm for first layer."
                )
                self.first_layer_norm = nn.Identity()
        else:
            self.first_layer_norm = nn.Identity()

        # --- Subsequent Hidden Layers (FF_Layer instances) ---
        # These map hidden_i -> hidden_{i+1}
        self.layers = nn.ModuleList()
        current_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:  # Iterate from the second hidden dim onwards
            self.layers.append(
                FF_Layer(
                    current_dim,
                    h_dim,
                    activation_cls=act_cls,
                    normalize=normalize_layers,
                    norm_type=norm_type,
                    bias=bias,
                    norm_eps=norm_eps,  # Pass epsilon
                )
            )
            current_dim = h_dim

        logger.info(
            f"Initialized FF_MLP with {len(hidden_dims)} effective hidden layers "
            f"(1 input adapter: {input_dim}->{hidden_dims[0]} + {len(self.layers)} FF_Layers)."
        )
        all_layer_dims_str = " -> ".join(map(str, [input_dim] + hidden_dims))
        logger.info(f"Layer dimensions: {all_layer_dims_str}")

    def forward_upto(
        self, x_flattened_modified: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """
        Performs a forward pass through layers 0 to layer_idx (inclusive), returning
        the normalized activation output of that layer.
        This is used during layer-wise training to get the input for the *next* layer.

        Args:
            x_flattened_modified (torch.Tensor): Flattened input tensor with embedded labels
                                                (shape: [batch_size, input_dim]).
            layer_idx (int): Index of the last *effective* hidden layer to run
                             (0 to num_hidden_layers - 1). Output of this layer is returned.

        Returns:
            torch.Tensor: Output activation tensor from the specified layer (normalized).
        """
        if not (0 <= layer_idx < len(self.hidden_dims)):
            raise ValueError(
                f"layer_idx {layer_idx} out of range for {len(self.hidden_dims)} effective hidden layers."
            )

        # --- Input Adapter Layer (Hidden Layer 0) ---
        # Input x is already flattened and modified
        current_activation = self.input_adapter_layer(x_flattened_modified)
        current_activation = self.first_layer_activation(current_activation)
        current_activation = self.first_layer_norm(
            current_activation
        )  # Output of layer 0 (normalized)

        if layer_idx == 0:
            return current_activation

        # --- Subsequent FF Layers ---
        # Run FF_Layers 0 to layer_idx-1 (corresponding to layers mapping H1->H2, ..., H_k -> H_{k+1})
        # The loop goes up to layer_idx-1 because self.layers[i] computes the output of hidden layer i+1.
        for i in range(layer_idx):
            if i >= len(self.layers):  # Safety check
                raise IndexError(
                    f"Trying to access self.layers[{i}] but only {len(self.layers)} layers exist."
                )
            # FF_Layer's forward method applies: Linear -> Activation -> Norm
            current_activation = self.layers[i](current_activation)

        return current_activation

    def forward_goodness_per_layer(
        self, x_flattened_modified: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Performs a forward pass calculating 'goodness' (sum of squared activations
        before normalization) for each effective hidden layer. Used for inference.

        Args:
            x_flattened_modified (torch.Tensor): Flattened input tensor with embedded labels.

        Returns:
            List[torch.Tensor]: List containing goodness scores (shape: [batch_size])
                                for each hidden layer (length = num_hidden_layers).
        """
        if x_flattened_modified.shape[1] != self.input_dim:
            raise ValueError(
                f"Input tensor dim {x_flattened_modified.shape[1]} != model input_dim {self.input_dim}"
            )

        layer_goodness = []

        # --- Input Adapter Layer (Hidden Layer 0) ---
        x_adapt_lin = self.input_adapter_layer(x_flattened_modified)
        x_adapt_act = self.first_layer_activation(x_adapt_lin)
        # Goodness calculation: Sum of squares BEFORE normalization
        goodness_0 = torch.sum(x_adapt_act.pow(2), dim=1)
        layer_goodness.append(goodness_0)
        # Pass the normalized activation to the next layer
        current_activation = self.first_layer_norm(x_adapt_act)

        # --- Subsequent FF Layers ---
        for layer in self.layers:  # These are FF_Layer instances
            # forward_with_goodness returns (normalized_output, goodness_before_norm)
            # We pass the normalized output ('current_activation') to the next layer
            # and collect the goodness calculated *within* that layer before its norm step.
            current_activation, goodness = layer.forward_with_goodness(
                current_activation
            )
            layer_goodness.append(goodness)

        if len(layer_goodness) != len(self.hidden_dims):
            logger.warning(
                f"Expected {len(self.hidden_dims)} goodness scores, got {len(layer_goodness)}"
            )

        return layer_goodness
