# File: src/architectures/ff_mlp.py
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

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        activation_cls (Type[nn.Module]): The activation function class (e.g., nn.ReLU).
        normalize (bool): Whether to apply normalization after activation. Defaults to True.
        norm_type (str): Type of normalization ('length' or 'layernorm'). Defaults to 'length'.
        bias (bool): Whether the linear layer uses a bias term. Defaults to True.
        norm_eps (float): Epsilon added to the denominator for numerical stability
                          during normalization (especially length normalization).
                          Defaults to 1e-5 for better stability.
        bias_init (float): Value to initialize bias terms. Defaults to 0.1 for ReLU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_cls: Type[nn.Module] = nn.ReLU,
        normalize: bool = True,
        norm_type: str = "length",
        bias: bool = True,
        norm_eps: float = 1e-5,
        bias_init: float = 0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation_cls()
        self.norm_eps = norm_eps

        # --- Explicit Initialization ---
        if isinstance(self.activation, nn.ReLU):
            nn.init.kaiming_uniform_(
                self.linear.weight, a=math.sqrt(5), nonlinearity="relu"
            )
        elif isinstance(self.activation, nn.Tanh):
            nn.init.xavier_uniform_(self.linear.weight)
        else:
            nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

        if bias and self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, bias_init)
            # logger.debug(f"FF_Layer: Initialized bias to {bias_init}") # Keep bias init logging less verbose

        self.norm_type = norm_type.lower()
        self.normalize = normalize
        if self.normalize:
            if self.norm_type == "layernorm":
                self.norm_layer = nn.LayerNorm(
                    out_features, elementwise_affine=True, eps=norm_eps
                )
                logger.debug(f"FF_Layer: Using nn.LayerNorm (eps={norm_eps}).")
            elif self.norm_type == "length":
                self.norm_layer = lambda x: x / (
                    torch.linalg.norm(x, dim=1, keepdim=True) + self.norm_eps
                )
                logger.debug(f"FF_Layer: Using Length Norm (eps={norm_eps}).")
            else:
                raise ValueError(f"Unsupported norm_type: {norm_type}.")
        else:
            self.norm_layer = nn.Identity()

        # logger.debug( # Keep constructor logging less verbose
        #     f"FF_Layer: In={in_features}, Out={out_features}, Normalize={normalize} "
        #     f"(Type: {self.norm_type if self.normalize else 'None'}, eps={self.norm_eps}), "
        #     f"Bias={bias}, Activation={activation_cls.__name__}"
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lin = self.linear(x)
        x_act = self.activation(x_lin)
        x_norm = self.norm_layer(x_act)
        return x_norm

    # --- MODIFIED: Added optional debug logging ---
    def forward_with_goodness(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_lin = self.linear(x)
        x_act = self.activation(x_lin)
        goodness = torch.sum(x_act.pow(2), dim=1)

        # <<< ADDED Debug Logging (sampled) >>>
        # Log ~1% of the time to avoid excessive output
        # Ensure self.training check isn't strictly necessary unless logging desired only during training
        if torch.randint(0, 100, (1,)).item() == 0:
             try:
                 with torch.no_grad():
                     # Calculate norms safely
                     x_act_norm_val = torch.linalg.norm(x_act, dim=1)
                     # Handle potential division by zero if batch size is 0 (shouldn't happen with DataLoader usually)
                     if x_act_norm_val.numel() > 0:
                        x_act_norm_mean = x_act_norm_val.mean().item()
                     else:
                        x_act_norm_mean = float('nan')

                     if goodness.numel() > 0:
                         goodness_mean = goodness.mean().item()
                     else:
                         goodness_mean = float('nan')

                     logger.debug(f"FF_Layer Debug: x_act L2 Norm (mean): {x_act_norm_mean:.4f}, Goodness (mean): {goodness_mean:.4f}")
             except Exception as e_debug:
                 # Catch errors during debug logging itself to prevent crashes
                 logger.warning(f"Error during FF_Layer debug logging: {e_debug}")
        # <<< END Debug Logging >>>


        x_norm = self.norm_layer(x_act)
        return x_norm, goodness


class FF_MLP(nn.Module):
    """
    Multi-Layer Perceptron specifically designed for the Forward-Forward algorithm.

    Args:
        input_dim (int): Dimensionality of the flattened input.
        hidden_dims (List[int]): List containing the number of neurons for each hidden layer.
        num_classes (int): Number of output classes.
        activation (str): Name of the activation function. Defaults to 'relu'.
        normalize_layers (bool): Whether to apply normalization. Defaults to True.
        norm_type (str): Type of normalization ('length' or 'layernorm'). Defaults to 'length'.
        bias (bool): Whether linear layers use bias terms. Defaults to True.
        norm_eps (float): Epsilon for normalization stability. Defaults to 1e-5.
        bias_init (float): Value to initialize bias terms. Defaults to 0.1 for ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = "relu",
        normalize_layers: bool = True,
        norm_type: str = "length",
        bias: bool = True,
        norm_eps: float = 1e-5,
        bias_init: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.num_hidden_layers = len(hidden_dims) # Store number of hidden layers for evaluation logic
        self.normalize_layers = normalize_layers
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.bias_init = bias_init

        if not hidden_dims:
            raise ValueError("hidden_dims list cannot be empty for FF_MLP.")

        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "tanh":
            act_cls = nn.Tanh
            if bias_init == 0.1:
                logger.info("Using Tanh activation, setting bias_init to 0.0.")
                self.bias_init = 0.0
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # --- Input Adapter Layer (Effective Hidden Layer 0) ---
        self.input_adapter_layer = nn.Linear(input_dim, hidden_dims[0], bias=bias)
        self.first_layer_activation = act_cls()

        # --- Explicit Initialization for input_adapter_layer ---
        if isinstance(self.first_layer_activation, nn.ReLU):
            nn.init.kaiming_uniform_(self.input_adapter_layer.weight, a=math.sqrt(5), nonlinearity="relu")
        elif isinstance(self.first_layer_activation, nn.Tanh):
            nn.init.xavier_uniform_(self.input_adapter_layer.weight)
        else:
            nn.init.kaiming_uniform_(self.input_adapter_layer.weight, a=math.sqrt(5))

        if bias and self.input_adapter_layer.bias is not None:
            nn.init.constant_(self.input_adapter_layer.bias, self.bias_init)
            # logger.debug(f"FF_MLP Input Adapter: Initialized bias to {self.bias_init}")

        # Apply normalization to the output of the first effective hidden layer
        if normalize_layers:
            if norm_type.lower() == "layernorm":
                self.first_layer_norm = nn.LayerNorm(hidden_dims[0], eps=self.norm_eps)
            elif norm_type.lower() == "length":
                self.first_layer_norm = lambda x: x / (torch.linalg.norm(x, dim=1, keepdim=True) + self.norm_eps)
            else:
                logger.warning(f"Invalid norm_type '{norm_type}', disabling norm for first layer.")
                self.first_layer_norm = nn.Identity()
        else:
            self.first_layer_norm = nn.Identity()

        # --- Subsequent Hidden Layers (FF_Layer instances) ---
        self.layers = nn.ModuleList()
        current_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            self.layers.append(
                FF_Layer(
                    current_dim,
                    h_dim,
                    activation_cls=act_cls,
                    normalize=normalize_layers,
                    norm_type=norm_type,
                    bias=bias,
                    norm_eps=self.norm_eps,
                    bias_init=self.bias_init,
                )
            )
            current_dim = h_dim

        logger.info(
            f"Initialized FF_MLP with {len(hidden_dims)} hidden layers. "
            f"Norm: {self.norm_type if self.normalize_layers else 'None'}, eps: {self.norm_eps}, Bias init: {self.bias_init}"
        )
        all_layer_dims_str = " -> ".join(map(str, [input_dim] + hidden_dims))
        logger.info(f"Layer dimensions: {all_layer_dims_str}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Basic forward pass for compatibility (e.g., FLOPs profiling)."""
        current_activation = self.input_adapter_layer(x)
        current_activation = self.first_layer_activation(current_activation)
        current_activation = self.first_layer_norm(current_activation)
        for layer in self.layers:
            current_activation = layer(current_activation)
        return current_activation # Returns final hidden layer output (normalized)

    def forward_upto(
        self, x_flattened_modified: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Forward pass up to layer_idx, returning normalized output."""
        if not (0 <= layer_idx < len(self.hidden_dims)):
            raise ValueError(f"layer_idx {layer_idx} out of range (0 to {len(self.hidden_dims)-1}).")
        current_activation = self.input_adapter_layer(x_flattened_modified)
        current_activation = self.first_layer_activation(current_activation)
        current_activation = self.first_layer_norm(current_activation)
        if layer_idx == 0: # layer_idx 0 corresponds to output of first hidden layer (after norm)
            return current_activation
        # Loop runs layer_idx times to get output of layer layer_idx+1 (e.g., layer_idx=1 runs self.layers[0])
        for i in range(layer_idx):
            if i >= len(self.layers):
                 raise IndexError(f"Internal index {i} exceeds number of subsequent layers ({len(self.layers)}) for requested layer_idx {layer_idx}.")
            current_activation = self.layers[i](current_activation)
        return current_activation

    def forward_goodness_per_layer(
        self, x_flattened_modified: torch.Tensor
    ) -> List[torch.Tensor]:
        """Forward pass calculating goodness per layer."""
        if x_flattened_modified.shape[1] != self.input_dim:
            raise ValueError(f"Input dim mismatch. Got {x_flattened_modified.shape[1]}, expected {self.input_dim}")
        layer_goodness = []
        # --- Layer 1 (Input Adapter Layer) ---
        x_adapt_lin = self.input_adapter_layer(x_flattened_modified)
        x_adapt_act = self.first_layer_activation(x_adapt_lin)
        goodness_0 = torch.sum(x_adapt_act.pow(2), dim=1)
        layer_goodness.append(goodness_0)
        current_activation = self.first_layer_norm(x_adapt_act) # Normalize for next layer's input

        # --- Subsequent Hidden Layers ---
        for layer in self.layers:
            # forward_with_goodness calculates goodness on pre-normalized activation
            current_activation, goodness = layer.forward_with_goodness(current_activation)
            layer_goodness.append(goodness)
            # Note: current_activation is now the normalized output, ready for the *next* layer

        if len(layer_goodness) != len(self.hidden_dims):
            logger.warning(f"Goodness scores length mismatch. Expected {len(self.hidden_dims)}, got {len(layer_goodness)}")
        return layer_goodness