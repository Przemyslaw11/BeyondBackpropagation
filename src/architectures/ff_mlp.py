# File: src/architectures/ff_mlp.py (MODIFIED - Replaces previous content)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math
import logging

from src.utils.metrics import calculate_accuracy # Import from your utils

logger = logging.getLogger(__name__)

# --- Custom ReLU with Gradient Passthrough ---
class ReLU_full_grad(torch.autograd.Function):
    """ReLU activation function that passes through the gradient irrespective of its input value."""
    @staticmethod
    def forward(ctx: Any, input_val: torch.Tensor) -> torch.Tensor:
        return input_val.clamp(min=0)
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clone()

# --- Modified FF_MLP to match Hinton's reference logic ---
class FF_MLP(torch.nn.Module):
    """
    MODIFIED MLP model to implement Hinton's reference Forward-Forward (FF) algorithm logic.
    Uses simultaneous local gradient updates via detach. Includes downstream classifier.
    Structure adapted to work with the modified training loop in src/algorithms/ff.py.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        activation: str = "relu",
        normalize_layers: bool = True, # Kept for potential future use, but ref code always normalizes
        norm_type: str = "length",     # Kept for potential future use, but ref code uses 'length'
        bias: bool = True,
        norm_eps: float = 1.0e-8,      # Default from ref code
        bias_init: float = 0.0,        # Default from ref code
        config: Optional[Dict[str, Any]] = None, # Accept full config for algo params
        device: Optional[torch.device] = None,   # Accept device
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.num_layers = len(hidden_dims)
        self.activation_name = activation.lower()
        self.use_bias = bias
        self.bias_init = bias_init
        self.norm_eps = norm_eps
        self.device = device if device else torch.device("cpu") # Default to CPU if not provided

        # --- Get Algorithm Params from Config ---
        # Use defaults matching reference if config is not provided
        algo_config = config.get("algorithm_params", {}) if config else {}
        self.threshold = float(algo_config.get("threshold", 2.0))
        self.peer_normalization_factor = float(algo_config.get("peer_normalization_factor", 0.0))
        self.use_peer_normalization = self.peer_normalization_factor > 0.0
        self.peer_momentum = float(algo_config.get("peer_momentum", 0.9))

        # --- Activation Function Selection ---
        if self.activation_name == 'relu':
            self.act_fn_train = ReLU_full_grad()
            self.act_fn_eval = nn.ReLU()
        elif self.activation_name == 'tanh':
            self.act_fn_train = nn.Tanh()
            self.act_fn_eval = nn.Tanh()
            if self.bias_init != 0.0:
                logger.info(f"{self.__class__.__name__}: Using Tanh activation, ensuring bias_init is 0.0 (was {self.bias_init}).")
                self.bias_init = 0.0
        else:
            logger.warning(f"{self.__class__.__name__}: Unsupported activation '{self.activation_name}'. Defaulting to ReLU.")
            self.act_fn_train = ReLU_full_grad()
            self.act_fn_eval = nn.ReLU()
            self.bias_init = 0.0 # Ensure default

        # --- FF Layers (nn.Linear) ---
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            linear_layer = nn.Linear(current_dim, h_dim, bias=self.use_bias)
            self._init_layer_weights(linear_layer) # Apply initialization
            self.layers.append(linear_layer)
            current_dim = h_dim

        # --- Losses ---
        self.ff_loss_criterion = nn.BCEWithLogitsLoss()
        self.classification_loss_criterion = nn.CrossEntropyLoss()

        # --- Peer Normalization ---
        if self.use_peer_normalization:
            self.running_means = [
                torch.zeros(h_dim, device=self.device) + 0.5
                for h_dim in self.hidden_dims
            ]
            logger.info(f"{self.__class__.__name__}: Using Peer Normalization (Factor: {self.peer_normalization_factor}, Momentum: {self.peer_momentum}).")
        else:
             logger.info(f"{self.__class__.__name__}: Peer Normalization disabled.")

        # --- Downstream Classification Head ---
        if self.num_layers <= 1:
            logger.warning(f"{self.__class__.__name__}: Only <=1 hidden layer. Downstream classifier input dim based on layer 0.")
            channels_for_classification_loss = self.hidden_dims[0]
        else:
            # Use activations from layers 1 to N-1 (indices 1 to num_layers-1)
            channels_for_classification_loss = sum(self.hidden_dims[1:]) # Sum dims from layer 1 onwards

        # Check if calculated dim is zero (can happen if hidden_dims[1:] is empty)
        if channels_for_classification_loss <= 0 and self.num_layers > 1:
             logger.warning(f"{self.__class__.__name__}: Downstream classifier input dimension calculated as {channels_for_classification_loss}. Check hidden_dims config.")
             # Fallback or error? Let's use layer 0 if needed.
             channels_for_classification_loss = self.hidden_dims[0] if self.num_layers > 0 else 0


        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, self.num_classes, bias=False)
        )
        self._init_classifier_weights()

        logger.info(
            f"Initialized Modified FF_MLP with {self.num_layers} hidden layers. "
            f"Input: {self.input_dim}, Hidden: {self.hidden_dims}, Classes: {self.num_classes}. "
            f"Activation: {self.activation_name}, Bias Init: {self.bias_init}. "
            f"FF Threshold: {self.threshold}. Norm Eps: {self.norm_eps}. "
            f"Downstream classifier input dim: {channels_for_classification_loss}."
        )

    def _init_layer_weights(self, layer: nn.Linear):
        """Initializes weights for a single FF linear layer."""
        if isinstance(self.act_fn_eval, nn.ReLU):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), nonlinearity='relu')
        elif isinstance(self.act_fn_eval, nn.Tanh):
            nn.init.xavier_uniform_(layer.weight)
        else: nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        if self.use_bias and layer.bias is not None:
            nn.init.constant_(layer.bias, self.bias_init)

    def _init_classifier_weights(self):
        """Initializes weights for the downstream linear classifier."""
        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear): nn.init.zeros_(m.weight)

    def _layer_norm(self, z: torch.Tensor) -> torch.Tensor:
        """Applies length normalization."""
        norm = torch.linalg.norm(z, dim=1, keepdim=True)
        return z / (norm + self.norm_eps)

    def _calc_peer_normalization_loss(self, layer_idx: int, z_pos: torch.Tensor) -> torch.Tensor:
        """Calculates peer normalization loss."""
        if not self.use_peer_normalization or layer_idx >= len(self.running_means):
            return torch.zeros(1, device=self.device)
        mean_activity = torch.mean(z_pos, dim=0)
        running_mean = self.running_means[layer_idx]
        new_running_mean = running_mean.detach() * self.peer_momentum + mean_activity * (1 - self.peer_momentum)
        self.running_means[layer_idx] = new_running_mean
        peer_loss = (torch.mean(new_running_mean) - new_running_mean) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z_pre_norm: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Calculates FF loss and accuracy."""
        sum_of_squares = torch.sum(z_pre_norm ** 2, dim=-1)
        logits = sum_of_squares - self.threshold
        ff_loss = self.ff_loss_criterion(logits, labels.float())
        with torch.no_grad():
            ff_accuracy = (torch.sum((torch.sigmoid(logits) > 0.5) == labels) / z_pre_norm.shape[0]).item() * 100.0
        return ff_loss, ff_accuracy

    # --- Standard Forward (for FLOPs/Compatibility) ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass using standard activation and normalization."""
        if x.shape[1] != self.input_dim:
            try: x = x.view(x.shape[0], self.input_dim)
            except RuntimeError: raise ValueError(f"Cannot reshape input {x.shape} to ({x.shape[0]}, {self.input_dim})")
        z = self._layer_norm(x) # Normalize input
        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            z_act = self.act_fn_eval(z_pre_act) # Use standard activation
            z = self._layer_norm(z_act) # Normalize for next layer
        return z # Return activation of last hidden layer

    # --- FF Training Forward ---
    def forward_ff_train(self, z_stacked: torch.Tensor, posneg_labels: torch.Tensor, current_batch_size: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Performs the forward pass for FF training."""
        scalar_outputs = {"Loss": torch.zeros(1, device=self.device), "Peer_Normalization_Loss_Total": torch.zeros(1, device=self.device), "FF_Loss_Total": torch.zeros(1, device=self.device)}
        z = z_stacked.reshape(z_stacked.shape[0], -1); z = self._layer_norm(z)
        normalized_activations_for_downstream = []
        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            z_act = self.act_fn_train.apply(z_pre_act) # Use ReLU_full_grad
            if self.use_peer_normalization:
                z_pos = z_act[:current_batch_size]
                peer_loss = self._calc_peer_normalization_loss(idx, z_pos)
                scalar_outputs["Peer_Normalization_Loss_Total"] += peer_loss
                scalar_outputs[f"Layer_{idx+1}/Peer_Norm_Loss"] = peer_loss.item()
                scalar_outputs["Loss"] += self.peer_normalization_factor * peer_loss
            ff_loss, ff_accuracy = self._calc_ff_loss(z_act, posneg_labels)
            scalar_outputs[f"Layer_{idx+1}/FF_Loss"] = ff_loss.item()
            scalar_outputs[f"Layer_{idx+1}/FF_Accuracy"] = ff_accuracy
            scalar_outputs["FF_Loss_Total"] += ff_loss; scalar_outputs["Loss"] += ff_loss
            z = z_act.detach(); z_norm = self._layer_norm(z)
            # Collect activations from layer 1 onwards for downstream classifier
            if idx >= 1:
                normalized_activations_for_downstream.append(z_norm[:current_batch_size].detach())
            z = z_norm
        if normalized_activations_for_downstream:
            try: input_classification_model = torch.cat(normalized_activations_for_downstream, dim=-1)
            except Exception as e_cat: logger.error(f"FF train cat error: {e_cat}"); input_classification_model = torch.zeros((current_batch_size, 0), device=self.device)
        else: input_classification_model = torch.zeros((current_batch_size, 0), device=self.device)
        self._current_downstream_input = input_classification_model
        if any(p.requires_grad for p in self.linear_classifier.parameters()): self._current_downstream_input.requires_grad_(True)
        return scalar_outputs["Loss"], scalar_outputs

    # --- Downstream Classifier Forward ---
    def forward_downstream_only(self, class_labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Performs the forward pass for the downstream classification model ONLY."""
        if not hasattr(self, "_current_downstream_input"): logger.error("Downstream missing input."); return torch.zeros(1, device=self.device, requires_grad=True), 0.0
        input_cls = self._current_downstream_input
        expected_dim = 0
        try: expected_dim = self.linear_classifier[0].in_features
        except (IndexError, AttributeError): logger.error("Could not get classifier input dim."); return torch.zeros(1, device=self.device, requires_grad=True), 0.0
        if expected_dim == 0: logger.debug("Downstream classifier expects 0 features."); dummy_loss = torch.zeros(1, device=self.device); return dummy_loss, 0.0
        if input_cls.shape[1] != expected_dim: logger.error(f"Downstream dim mismatch: Input {input_cls.shape[1]}, Expected {expected_dim}."); dummy_loss = torch.zeros(1, device=self.device); return dummy_loss, 0.0

        output = self.linear_classifier(input_cls)
        classification_loss = self.classification_loss_criterion(output, class_labels)
        with torch.no_grad(): classification_accuracy = calculate_accuracy(output.data, class_labels)
        if hasattr(self, "_current_downstream_input"): del self._current_downstream_input
        return classification_loss, classification_accuracy

    # --- Evaluation Forward (Goodness) ---
    def forward_goodness_per_layer(self, x_flattened_modified: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass calculating goodness per layer for evaluation."""
        if x_flattened_modified.shape[1] != self.input_dim: raise ValueError(f"Eval input dim mismatch: {x_flattened_modified.shape[1]} vs {self.input_dim}")
        layer_goodness = []; z = x_flattened_modified.reshape(x_flattened_modified.shape[0], -1); z = self._layer_norm(z)
        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            z_act = self.act_fn_eval(z_pre_act) # Use standard activation for eval
            goodness = torch.sum(z_act.pow(2), dim=1); layer_goodness.append(goodness)
            z = self._layer_norm(z_act.detach())
        if len(layer_goodness) != len(self.hidden_dims): logger.warning(f"Eval goodness length mismatch: {len(layer_goodness)} vs {len(self.hidden_dims)}.")
        return layer_goodness