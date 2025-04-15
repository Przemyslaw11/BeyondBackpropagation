# File: ./src/architectures/ff_mlp.py # <<< MODIFIED >>>
# --------------------------------------------------------------------------------
# File: ./src/architectures/ff_mlp.py (FINAL CORRECTED - Matches Reference Behavior)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math
import logging

from src.utils.metrics import calculate_accuracy # Import from your utils

logger = logging.getLogger(__name__)

# --- Custom ReLU with Gradient Passthrough ( Matches Reference ) ---
class ReLU_full_grad(torch.autograd.Function):
    """ReLU activation function that passes through the gradient irrespective of its input value."""
    @staticmethod
    def forward(ctx: Any, input_val: torch.Tensor) -> torch.Tensor:
        return input_val.clamp(min=0)
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        # <<< CORRECTION: Pass gradient through unchanged. >>>
        return grad_output.clone()

# --- Modified FF_MLP to match Hinton's reference logic ---
class FF_MLP(torch.nn.Module):
    """
    CORRECTED MLP model implementing Hinton's reference Forward-Forward (FF) algorithm logic.
    Uses simultaneous local gradient updates via detach. Includes downstream classifier.
    Structure adapted to work with the modified training loop in src/algorithms/ff.py.
    """
    def __init__(
        self,
        # <<< MODIFICATION: Accept full config and device directly >>>
        config: Dict[str, Any],
        device: torch.device,
        # Keep individual params for potential direct instantiation / BP baseline adaptation
        input_dim: Optional[int] = None, # Now optional, inferred from config if needed
        hidden_dims: Optional[List[int]] = None,
        num_classes: Optional[int] = None,
        activation: Optional[str] = "relu",
        bias: Optional[bool] = True,
        norm_eps: Optional[float] = 1.0e-8,
        bias_init: Optional[float] = 0.0, # Note: Will be forced to 0.0 if activation is ReLU
    ):
        super().__init__()

        # Store config sections and device
        self.config = config
        self.model_config = config.get("model", {})
        self.model_params = self.model_config.get("params", {})
        self.algo_params = config.get("algorithm_params", {})
        self.data_config = config.get("data", {})
        self.loader_config = config.get("data_loader", {})
        self.device = device

        # --- Determine Architecture Params (Use config first, then args) ---
        if input_dim is None: # Infer input_dim if not provided
             input_channels = self.data_config.get("input_channels", 1)
             image_size = self.data_config.get("image_size", 28)
             self.input_dim = input_channels * image_size * image_size
        else: self.input_dim = input_dim

        self.hidden_dims = hidden_dims if hidden_dims is not None else self.model_params.get("hidden_dims", [1000, 1000, 1000])
        self.num_classes = num_classes if num_classes is not None else self.data_config.get("num_classes", 10)
        self.num_layers = len(self.hidden_dims)
        # <<< CORRECTION: Read activation from model_params >>>
        self.activation_name = (activation if activation is not None else self.model_params.get("activation", "ReLU")).lower()
        self.use_bias = bias if bias is not None else self.model_params.get("bias", True)
        # <<< CORRECTION: Read bias_init from model_params, but default to 0.0 if ReLU >>>
        self.bias_init = bias_init if bias_init is not None else self.model_params.get("bias_init", 0.0)
        # <<< CORRECTION: Read norm_eps from model_params >>>
        self.norm_eps = norm_eps if norm_eps is not None else self.model_params.get("norm_eps", 1.0e-8) # Default 1e-8

        # --- Get Algorithm Params from Config ---
        # <<< CORRECTION: Threshold removed from algo_params, calculated dynamically >>>
        # self.threshold = float(self.algo_params.get("threshold", 2.0)) # <<< REMOVED >>>
        self.peer_normalization_factor = float(self.algo_params.get("peer_normalization_factor", 0.0))
        self.use_peer_normalization = self.peer_normalization_factor > 0.0
        self.peer_momentum = float(self.algo_params.get("peer_momentum", 0.9))

        # --- Activation Function Selection ---
        # <<< CORRECTION: Ensure correct train/eval activations & bias init for ReLU >>>
        if self.activation_name == 'relu':
            self.act_fn_train = ReLU_full_grad() # Use special ReLU for FF training grads
            self.act_fn_eval = nn.ReLU()         # Use standard ReLU for goodness eval
            # <<< CORRECTION: Reference uses 0.0 bias init >>>
            if self.bias_init != 0.0:
                 logger.info(f"{self.__class__.__name__}: Using ReLU activation, forcing bias_init to 0.0 (was {self.bias_init}).")
                 self.bias_init = 0.0
        elif self.activation_name == 'tanh':
             logger.warning("Tanh activation specified, reference uses ReLU. Behavior might differ.")
             self.act_fn_train = nn.Tanh()
             self.act_fn_eval = nn.Tanh()
             # <<< CORRECTION: Ensure bias is 0.0 for Tanh too, as reference did for ReLU >>>
             if self.bias_init != 0.0:
                 logger.info(f"{self.__class__.__name__}: Using Tanh activation, forcing bias_init to 0.0 (was {self.bias_init}).")
                 self.bias_init = 0.0
        else:
            logger.error(f"{self.__class__.__name__}: Unsupported activation '{self.activation_name}'. Defaulting to ReLU.")
            self.act_fn_train = ReLU_full_grad()
            self.act_fn_eval = nn.ReLU()
            self.bias_init = 0.0 # Ensure default 0.0 bias init

        # --- FF Layers (nn.Linear) ---
        self.layers = nn.ModuleList()
        current_dim = self.input_dim
        for i, h_dim in enumerate(self.hidden_dims):
            linear_layer = nn.Linear(current_dim, h_dim, bias=self.use_bias)
            # <<< CORRECTION: Call corrected initialization >>>
            self._init_layer_weights(linear_layer)
            self.layers.append(linear_layer)
            current_dim = h_dim

        # --- Losses ---
        self.ff_loss_criterion = nn.BCEWithLogitsLoss()
        self.classification_loss_criterion = nn.CrossEntropyLoss()

        # --- Peer Normalization ---
        if self.use_peer_normalization:
            self.running_means = [
                torch.zeros(h_dim, device=self.device) + 0.5 # Initialize at 0.5 like ref
                for h_dim in self.hidden_dims
            ]
            logger.info(f"{self.__class__.__name__}: Using Peer Normalization (Factor: {self.peer_normalization_factor}, Momentum: {self.peer_momentum}).")
        else:
             logger.info(f"{self.__class__.__name__}: Peer Normalization disabled.")

        # --- Downstream Classification Head ---
        # <<< CORRECTION: Reference uses activations from layers 1..N-1 >>>
        # Layer indices: 0=L1, 1=L2, ..., N-1=LN. Collect from idx=1 onwards.
        if self.num_layers <= 1:
             logger.warning(f"{self.__class__.__name__}: Only <=1 hidden layer. Downstream classifier input uses layer 0 activation (size {self.hidden_dims[0] if self.num_layers > 0 else 0}).")
             # <<< CORRECTION: If only 1 layer (N=1), collect from layer 0 (idx=0) instead of 1..0 >>>
             channels_for_classification_loss = self.hidden_dims[0] if self.num_layers > 0 else 0
        else:
             # Sum dims from layer 1 (idx=1) up to layer N-1 (idx=N-1)
             channels_for_classification_loss = sum(self.hidden_dims[1:]) # Sum dims from layer 1 onwards

        if channels_for_classification_loss <= 0 and self.num_layers > 0:
             logger.error(f"{self.__class__.__name__}: Downstream classifier input dimension calculated as 0 or less ({channels_for_classification_loss}). Check hidden_dims config.")
             channels_for_classification_loss = self.hidden_dims[0] if self.num_layers > 0 else 0 # Fallback to first layer?
             if channels_for_classification_loss <= 0: raise ValueError("Cannot determine downstream classifier input dimension.")

        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, self.num_classes, bias=False) # No bias in ref classifier
        )
        # <<< CORRECTION: Call corrected initialization >>>
        self._init_classifier_weights()

        logger.info(
            f"Initialized Modified FF_MLP with {self.num_layers} hidden layers. "
            f"Input: {self.input_dim}, Hidden: {self.hidden_dims}, Classes: {self.num_classes}. "
            f"Activation: {self.activation_name}, Bias Init: {self.bias_init}. "
            f"Peer Norm: {self.use_peer_normalization}. Norm Eps: {self.norm_eps}. "
            f"Downstream classifier input dim: {channels_for_classification_loss}."
        )

    def _init_layer_weights(self, layer: nn.Linear):
        """Initializes weights for a single FF linear layer according to reference."""
        # <<< CORRECTION: Match reference normal initialization std=1/sqrt(fan_out) >>>
        # NOTE: Reference used shape[0] which is fan_out for nn.Linear
        std_dev = 1.0 / math.sqrt(layer.weight.shape[0])
        nn.init.normal_(layer.weight, mean=0, std=std_dev)
        # <<< CORRECTION: Match reference bias initialization (0.0 for ReLU) >>>
        if self.use_bias and layer.bias is not None:
            nn.init.constant_(layer.bias, self.bias_init) # self.bias_init is now 0.0 if ReLU

    def _init_classifier_weights(self):
        """Initializes weights for the downstream linear classifier."""
        # <<< CORRECTION: Match reference zero initialization >>>
        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear): nn.init.zeros_(m.weight)

    def _layer_norm(self, z: torch.Tensor) -> torch.Tensor:
        """Applies RMS length normalization matching reference code."""
        # <<< CORRECTION: Use RMS normalization like reference code >>>
        rms_norm = torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True))
        return z / (rms_norm + self.norm_eps)

    def _calc_peer_normalization_loss(self, layer_idx: int, z_pos: torch.Tensor) -> torch.Tensor:
        """Calculates peer normalization loss matching reference."""
        if not self.use_peer_normalization or layer_idx >= len(self.running_means):
            return torch.zeros(1, device=self.device)
        mean_activity = torch.mean(z_pos, dim=0)
        running_mean = self.running_means[layer_idx]
        new_running_mean = running_mean.detach() * self.peer_momentum + mean_activity * (1 - self.peer_momentum)
        self.running_means[layer_idx] = new_running_mean
        # <<< CORRECTION: Reference peer loss formula from comment >>>
        peer_loss = (torch.mean(new_running_mean) - new_running_mean) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z_pre_norm: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Calculates FF loss and accuracy using DYNAMIC threshold matching reference."""
        sum_of_squares = torch.sum(z_pre_norm ** 2, dim=-1)
        # <<< CORRECTION: Use dynamic threshold based on layer's output dimension >>>
        dynamic_threshold = float(z_pre_norm.shape[1]) # Threshold = number of neurons in the layer
        logits = sum_of_squares - dynamic_threshold
        ff_loss = self.ff_loss_criterion(logits, labels.float())
        with torch.no_grad():
            # Accuracy calculation requires sigmoid output compared to 0.5
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
        # Return activation of last hidden layer (consistent behavior for profiling)
        return z

    # --- FF Training Forward (Apply Corrections) ---
    def forward_ff_train(self, z_stacked: torch.Tensor, posneg_labels: torch.Tensor, current_batch_size: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Performs the forward pass for FF training, calculating local losses."""
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.device), # Combined loss to be returned
            "Peer_Normalization_Loss_Total": torch.zeros(1, device=self.device),
            "FF_Loss_Total": torch.zeros(1, device=self.device)
        }
        # Flatten and Normalize Input
        z = z_stacked.reshape(z_stacked.shape[0], -1)
        z = self._layer_norm(z)

        normalized_activations_for_downstream = []

        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            # <<< CORRECTION: Use custom ReLU_full_grad for FF training gradient >>>
            z_act = self.act_fn_train.apply(z_pre_act)

            # --- Peer Normalization ---
            if self.use_peer_normalization:
                z_pos = z_act[:current_batch_size]
                # <<< CORRECTION: Use corrected peer loss calc >>>
                peer_loss = self._calc_peer_normalization_loss(idx, z_pos)
                scalar_outputs["Peer_Normalization_Loss_Total"] += peer_loss
                scalar_outputs[f"Layer_{idx+1}/Peer_Norm_Loss"] = peer_loss.item() # Log per-layer loss
                # Add weighted peer loss to combined loss
                scalar_outputs["Loss"] += self.peer_normalization_factor * peer_loss

            # --- Forward-Forward Loss ---
            # <<< CORRECTION: Use corrected FF loss calc (dynamic threshold) >>>
            ff_loss, ff_accuracy = self._calc_ff_loss(z_act, posneg_labels)
            scalar_outputs[f"Layer_{idx+1}/FF_Loss"] = ff_loss.item()
            scalar_outputs[f"Layer_{idx+1}/FF_Accuracy"] = ff_accuracy
            scalar_outputs["FF_Loss_Total"] += ff_loss # Accumulate total FF loss
            # Add FF loss to combined loss
            scalar_outputs["Loss"] += ff_loss

            # --- Prepare Input for Next Layer ---
            # <<< CORRECTION: Detach *before* normalization for next layer >>>
            z_detached = z_act.detach()
            z_norm = self._layer_norm(z_detached) # Use corrected norm

            # <<< CORRECTION: Collect activations from layers 1..N-1 (or L0 if N=1) for downstream >>>
            # Layer indices: 0=L1, 1=L2, ..., N-1=LN.
            # If N=1, collect from idx=0.
            # If N > 1, collect from idx=1 onwards.
            should_collect_for_downstream = (idx == 0 and self.num_layers == 1) or (idx >= 1)
            if should_collect_for_downstream:
                # We need the normalized, detached activations of the *positive* samples only
                normalized_activations_for_downstream.append(z_norm[:current_batch_size].detach())

            # <<< CORRECTION: Input for next layer is normalized & detached activation >>>
            z = z_norm

        # --- Assemble Downstream Classifier Input ---
        if normalized_activations_for_downstream:
            try:
                input_classification_model = torch.cat(normalized_activations_for_downstream, dim=-1)
            except Exception as e_cat:
                logger.error(f"FF train cat error: {e_cat}")
                # Fallback: create empty tensor with correct batch size
                input_classification_model = torch.zeros((current_batch_size, 0), device=self.device)
        else:
             # This case should only happen if num_layers is 0, which isn't supported
             logger.error("FF_MLP: No activations collected for downstream classifier, unexpected state.")
             input_classification_model = torch.zeros((current_batch_size, 0), device=self.device)


        # Store for separate downstream forward pass
        self._current_downstream_input = input_classification_model
        # <<< CORRECTION: Ensure requires_grad is set if classifier is trainable >>>
        if any(p.requires_grad for p in self.linear_classifier.parameters()):
             # Ensure grad flows if needed (important for SGD group setup)
             self._current_downstream_input = self._current_downstream_input.clone().requires_grad_(True)
        else:
             # Detach if classifier isn't being trained
             self._current_downstream_input = self._current_downstream_input.detach()


        # Return the combined FF + Peer Norm loss and the full metrics dict
        return scalar_outputs["Loss"], scalar_outputs

    # --- Downstream Classifier Forward (Apply Corrections) ---
    def forward_downstream_only(self, class_labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Performs the forward pass for the downstream classification model ONLY."""
        if not hasattr(self, "_current_downstream_input"):
            logger.error("Downstream forward called before main FF forward pass. Input is missing.")
            # Return dummy loss/acc. Ensure loss requires grad if classifier does.
            dummy_loss = torch.zeros(1, device=self.device)
            if any(p.requires_grad for p in self.linear_classifier.parameters()):
                 dummy_loss.requires_grad_(True)
            return dummy_loss, 0.0

        input_cls = self._current_downstream_input
        expected_dim = 0
        try:
            # <<< CORRECTION: Access Linear layer inside Sequential >>>
            expected_dim = self.linear_classifier[0].in_features
        except (IndexError, AttributeError):
            logger.error("Could not get downstream classifier input dimension.")
            dummy_loss = torch.zeros(1, device=self.device)
            if any(p.requires_grad for p in self.linear_classifier.parameters()):
                 dummy_loss.requires_grad_(True)
            return dummy_loss, 0.0

        # Handle case where expected dim is 0 (e.g., if collection failed)
        if expected_dim == 0:
            # This might happen if num_layers=1 and collection logic was strictly L1..N-1
            # Now handled by collecting L0 if N=1. If it still happens, log warning.
            logger.warning("Downstream classifier expects 0 features. Returning zero loss/acc.")
            dummy_loss = torch.zeros(1, device=self.device)
            if any(p.requires_grad for p in self.linear_classifier.parameters()):
                 dummy_loss.requires_grad_(True)
            return dummy_loss, 0.0

        # Check for dim mismatch AFTER checking expected_dim > 0
        if input_cls.shape[1] != expected_dim:
             logger.error(f"Downstream dimension mismatch: Input has {input_cls.shape[1]}, classifier expects {expected_dim}.")
             dummy_loss = torch.zeros(1, device=self.device)
             if any(p.requires_grad for p in self.linear_classifier.parameters()):
                 dummy_loss.requires_grad_(True)
             return dummy_loss, 0.0

        # Proceed if dimensions match
        output = self.linear_classifier(input_cls)
        classification_loss = self.classification_loss_criterion(output, class_labels)
        with torch.no_grad():
            classification_accuracy = calculate_accuracy(output.data, class_labels)

        # Clean up stored input (optional, helps memory slightly)
        # if hasattr(self, "_current_downstream_input"):
        #     del self._current_downstream_input

        return classification_loss, classification_accuracy

    # --- Evaluation Forward (Goodness - Apply Corrections) ---
    def forward_goodness_per_layer(self, x_flattened_modified: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass calculating goodness per layer for evaluation."""
        if x_flattened_modified.shape[1] != self.input_dim:
            raise ValueError(f"Eval input dim mismatch: {x_flattened_modified.shape[1]} vs {self.input_dim}")

        layer_goodness = []
        z = x_flattened_modified.reshape(x_flattened_modified.shape[0], -1)
        z = self._layer_norm(z) # Normalize input

        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            # <<< CORRECTION: Use standard activation (act_fn_eval) for eval goodness >>>
            z_act = self.act_fn_eval(z_pre_act)
            goodness = torch.sum(z_act.pow(2), dim=1)
            layer_goodness.append(goodness)
            # <<< CORRECTION: Detach before normalization for next layer input, matching train logic >>>
            z_detached = z_act.detach()
            z = self._layer_norm(z_detached) # Use corrected norm

        if len(layer_goodness) != len(self.hidden_dims):
            logger.warning(f"Eval goodness length mismatch: {len(layer_goodness)} vs {len(self.hidden_dims)}.")
        return layer_goodness