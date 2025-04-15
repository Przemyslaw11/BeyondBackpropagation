# File: src/architectures/ff_hinton.py
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
        # Pass gradient through unchanged.
        return grad_output.clone()

# --- Hinton-style FF MLP Model ---
class FF_Hinton_MLP(torch.nn.Module):
    """
    MLP model adapted from Hinton's reference code for the Forward-Forward (FF) algorithm.
    Uses simultaneous local gradient updates via detach. Includes downstream classifier.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ):
        super().__init__()

        # Store config sections and device
        self.model_config = config.get("model", {})
        self.model_params = self.model_config.get("params", {})
        self.algo_params = config.get("algorithm_params", {})
        self.data_config = config.get("data", {})
        self.loader_config = config.get("data_loader", {})
        self.device = device

        # --- Model Architecture ---
        self.input_dim = self.model_params.get("input_dim", 784) # Get from params
        self.hidden_dims = self.model_params.get("hidden_dims", [2000, 2000])
        self.num_classes = self.data_config.get("num_classes", 10)
        self.num_layers = len(self.hidden_dims)
        self.activation_name = self.model_params.get("activation", "ReLU").lower()
        self.use_bias = self.model_params.get("bias", True)
        self.bias_init = self.model_params.get("bias_init", 0.0) # Using 0.0 from ref code
        self.norm_eps = self.model_params.get("norm_eps", 1e-8) # Using 1e-8 from ref code

        if self.activation_name == 'relu':
            self.act_fn_train = ReLU_full_grad() # Use special ReLU for training
            self.act_fn_eval = nn.ReLU()         # Use standard ReLU for standard forward/eval
            # Bias init defaults to 0.0 if ReLU
        elif self.activation_name == 'tanh':
            self.act_fn_train = nn.Tanh()
            self.act_fn_eval = nn.Tanh()
            if self.bias_init != 0.0:
                 logger.info(f"FF_Hinton_MLP: Using Tanh activation, ensuring bias_init is 0.0 (was {self.bias_init}).")
                 self.bias_init = 0.0
        else:
            logger.error(f"FF_Hinton_MLP: Unsupported activation '{self.activation_name}'. Defaulting to ReLU.")
            self.act_fn_train = ReLU_full_grad()
            self.act_fn_eval = nn.ReLU()
            self.bias_init = self.model_params.get("bias_init", 0.0) # Ensure default

        # Initialize the model layers
        self.layers = nn.ModuleList()
        current_dim = self.input_dim
        for i, h_dim in enumerate(self.hidden_dims):
            linear_layer = nn.Linear(current_dim, h_dim, bias=self.use_bias)
            self._init_layer_weights(linear_layer) # Apply initialization
            self.layers.append(linear_layer)
            current_dim = h_dim

        # --- Losses ---
        self.ff_loss_criterion = nn.BCEWithLogitsLoss()
        self.classification_loss_criterion = nn.CrossEntropyLoss()
        self.threshold = float(self.algo_params.get("threshold", 2.0)) # Ensure float

        # --- Peer Normalization ---
        self.peer_normalization_factor = float(self.algo_params.get("peer_normalization_factor", 0.0)) # Ensure float
        self.use_peer_normalization = self.peer_normalization_factor > 0.0
        self.peer_momentum = float(self.algo_params.get("peer_momentum", 0.9)) # Ensure float
        if self.use_peer_normalization:
            self.running_means = [
                torch.zeros(h_dim, device=self.device) + 0.5
                for h_dim in self.hidden_dims
            ]
            logger.info(f"FF_Hinton_MLP: Using Peer Normalization (Factor: {self.peer_normalization_factor}, Momentum: {self.peer_momentum}).")
        else:
             logger.info("FF_Hinton_MLP: Peer Normalization disabled.")

        # --- Downstream Classification Head ---
        if self.num_layers <= 1:
            logger.warning("FF_Hinton_MLP: Only 1 hidden layer. Downstream classifier will have limited input (only layer 0).")
            channels_for_classification_loss = self.hidden_dims[0] # Use first hidden layer for 1-layer net
        else:
            channels_for_classification_loss = sum(self.hidden_dims[1:]) # Sum dims from layer 1 onwards

        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, self.num_classes, bias=False)
        )
        self._init_classifier_weights() # Initialize classifier weights

        logger.info(
            f"Initialized FF_Hinton_MLP with {self.num_layers} hidden layers. "
            f"Input: {self.input_dim}, Hidden: {self.hidden_dims}, Classes: {self.num_classes}. "
            f"Activation: {self.activation_name}, Bias Init: {self.bias_init}. "
            f"FF Threshold: {self.threshold}. Norm Eps: {self.norm_eps}. "
            f"Downstream classifier input dim: {channels_for_classification_loss}."
        )

    def _init_layer_weights(self, layer: nn.Linear):
        """Initializes weights for a single FF linear layer."""
        # Weight Initialization
        if isinstance(self.act_fn_eval, nn.ReLU):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), nonlinearity='relu')
        elif isinstance(self.act_fn_eval, nn.Tanh):
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        # Bias Initialization
        if self.use_bias and layer.bias is not None:
            nn.init.constant_(layer.bias, self.bias_init)

    def _init_classifier_weights(self):
        """Initializes weights for the downstream linear classifier."""
        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight) # Zero initialization as per reference code

    def _layer_norm(self, z: torch.Tensor) -> torch.Tensor:
        """Applies length normalization (dividing by L2 norm)."""
        norm = torch.linalg.norm(z, dim=1, keepdim=True)
        return z / (norm + self.norm_eps)

    def _calc_peer_normalization_loss(self, layer_idx: int, z_pos: torch.Tensor) -> torch.Tensor:
        """Calculates peer normalization loss for positive samples."""
        if not self.use_peer_normalization or layer_idx >= len(self.running_means):
            return torch.zeros(1, device=self.device)

        mean_activity = torch.mean(z_pos, dim=0)
        running_mean = self.running_means[layer_idx]
        # In-place update of running_mean might cause issues with graph if not careful
        # Assigning to self.running_means[idx] is generally safer
        new_running_mean = running_mean.detach() * self.peer_momentum + mean_activity * (1 - self.peer_momentum)
        self.running_means[layer_idx] = new_running_mean

        peer_loss = (torch.mean(new_running_mean) - new_running_mean) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(
        self, z_pre_norm: torch.Tensor, labels: torch.Tensor # labels: 0 for neg, 1 for pos
    ) -> Tuple[torch.Tensor, float]:
        """Calculates the FF loss and accuracy for a layer's activations (before normalization)."""
        sum_of_squares = torch.sum(z_pre_norm ** 2, dim=-1)
        logits = sum_of_squares - self.threshold # Use threshold here
        ff_loss = self.ff_loss_criterion(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels) / z_pre_norm.shape[0]
            ).item() * 100.0
        return ff_loss, ff_accuracy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for compatibility with tools like FLOPs profiler.
        Uses standard activation and normalization. Does NOT compute FF loss.
        Input x should already be flattened.
        """
        if x.shape[1] != self.input_dim:
             logger.warning(f"Standard forward expected input dim {self.input_dim}, got {x.shape[1]}. Reshaping.")
             # Attempt to reshape, assuming batch dim is first
             try:
                 x = x.view(x.shape[0], self.input_dim)
             except RuntimeError:
                 raise ValueError(f"Cannot reshape input of shape {x.shape} to ({x.shape[0]}, {self.input_dim})")

        # Normalize input first, consistent with ff_train forward
        z = self._layer_norm(x)

        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            z_act = self.act_fn_eval(z_pre_act) # Use standard activation
            z = self._layer_norm(z_act) # Normalize output

        # Standard forward doesn't typically return classifier output unless requested
        # This output 'z' is the normalized activation of the last hidden layer
        return z


    def forward_ff_train(
        self,
        z_stacked: torch.Tensor, # Concatenated pos and neg samples (Batch*2, Features)
        posneg_labels: torch.Tensor, # Labels: 1 for pos, 0 for neg (Batch*2)
        current_batch_size: int, # Original batch size (before concat)
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Performs the forward pass for FF training, calculating local losses.
        Returns the total combined loss for the batch and a dictionary of metrics.
        (Previously named 'forward')
        """
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.device),
            "Peer_Normalization_Loss_Total": torch.zeros(1, device=self.device),
            "FF_Loss_Total": torch.zeros(1, device=self.device),
        }

        z = z_stacked.reshape(z_stacked.shape[0], -1) # Ensure flattened input
        z = self._layer_norm(z) # Normalize input

        normalized_activations_for_downstream = []

        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            # Use special ReLU with grad passthrough for FF training
            z_act = self.act_fn_train.apply(z_pre_act)

            # --- Peer Normalization (using positive samples only) ---
            if self.use_peer_normalization:
                z_pos = z_act[:current_batch_size]
                peer_loss = self._calc_peer_normalization_loss(idx, z_pos)
                scalar_outputs["Peer_Normalization_Loss_Total"] += peer_loss
                scalar_outputs[f"Layer_{idx+1}/Peer_Norm_Loss"] = peer_loss.item()
                scalar_outputs["Loss"] += self.peer_normalization_factor * peer_loss

            # --- Forward-Forward Loss ---
            ff_loss, ff_accuracy = self._calc_ff_loss(z_act, posneg_labels)
            scalar_outputs[f"Layer_{idx+1}/FF_Loss"] = ff_loss.item()
            scalar_outputs[f"Layer_{idx+1}/FF_Accuracy"] = ff_accuracy
            scalar_outputs["FF_Loss_Total"] += ff_loss
            scalar_outputs["Loss"] += ff_loss

            # --- Prepare for next layer ---
            z = z_act.detach() # Detach before normalization for next layer
            z_norm = self._layer_norm(z)

            # Store normalized activations (excluding layer 0) for downstream task
            # Reference code uses layers 1..N-1.
            # So if num_layers = 4 (idx 0,1,2,3), we use activations from idx=1, 2
            if idx >= 1: # Store from layer 1 onwards
                 # Detach again just to be safe before appending
                normalized_activations_for_downstream.append(z_norm[:current_batch_size].detach())

            # Input for the next layer is the normalized detached activation
            z = z_norm

        # --- Combine activations for the linear classifier ---
        # Check if the list is non-empty before concatenating
        if normalized_activations_for_downstream:
            try:
                input_classification_model = torch.cat(normalized_activations_for_downstream, dim=-1)
                logger.debug(f"FF forward: Concatenated {len(normalized_activations_for_downstream)} layer activations for downstream. Shape: {input_classification_model.shape}")
            except Exception as e_cat:
                logger.error(f"FF forward: Error concatenating activations for downstream classifier: {e_cat}")
                # Create empty tensor with correct batch size but 0 features
                input_classification_model = torch.zeros((current_batch_size, 0), device=self.device)
        else:
            # Handle cases with <= 1 hidden layer or if list remained empty unexpectedly
            input_classification_model = torch.zeros((current_batch_size, 0), device=self.device)
            logger.debug("FF forward: No activations collected/concatenated for downstream classifier.")


        # Store classifier input for separate forward call
        # Ensure it requires grad if the classifier weights require grad (which they should)
        self._current_downstream_input = input_classification_model
        # Check if classifier has parameters requiring grad
        if any(p.requires_grad for p in self.linear_classifier.parameters()):
             self._current_downstream_input.requires_grad_(True) # Allow grad flow to classifier

        return scalar_outputs["Loss"], scalar_outputs

    def forward_downstream_only(self, class_labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Performs the forward pass for the downstream classification model ONLY.
        Requires `forward_ff_train` to have been called first.
        """
        if not hasattr(self, "_current_downstream_input"):
             logger.error("Downstream forward called before main FF forward pass. Input is missing.")
             return torch.zeros(1, device=self.device, requires_grad=True), 0.0 # Return dummy loss

        input_cls = self._current_downstream_input

        # Check dimensions BEFORE passing to classifier
        expected_dim = 0
        try:
            # Access the Linear layer inside the Sequential
            expected_dim = self.linear_classifier[0].in_features
        except (IndexError, AttributeError):
             logger.error("Could not determine expected input dimension for linear classifier.")
             return torch.zeros(1, device=self.device, requires_grad=True), 0.0

        # Handle case where classifier expects 0 features (e.g., <=1 hidden layer)
        if expected_dim == 0:
             logger.debug("Downstream forward: Classifier expects 0 input features. Returning zero loss/acc.")
             # Ensure the dummy loss requires grad if needed by the optimizer setup
             dummy_loss = torch.zeros(1, device=self.device)
             if any(p.requires_grad for p in self.linear_classifier.parameters()):
                 dummy_loss.requires_grad_(True)
             return dummy_loss, 0.0

        # Check for dimension mismatch
        if input_cls.shape[1] != expected_dim:
             logger.error(f"Downstream forward: Input dimension mismatch. Input has {input_cls.shape[1]}, classifier expects {expected_dim}.")
             # Return dummy loss/acc
             dummy_loss = torch.zeros(1, device=self.device)
             if any(p.requires_grad for p in self.linear_classifier.parameters()):
                 dummy_loss.requires_grad_(True)
             return dummy_loss, 0.0

        # Proceed if dimensions match
        output = self.linear_classifier(input_cls)
        classification_loss = self.classification_loss_criterion(output, class_labels)

        with torch.no_grad():
             classification_accuracy = calculate_accuracy(output.data, class_labels)

        # Clean up stored input
        if hasattr(self, "_current_downstream_input"):
             del self._current_downstream_input

        return classification_loss, classification_accuracy

    def forward_goodness_per_layer(
        self, x_flattened_modified: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Forward pass calculating goodness per layer (for evaluation).
        Input should be the flattened image with an embedded label.
        Uses standard ReLU for activation during goodness calculation.
        """
        if x_flattened_modified.shape[1] != self.input_dim:
            raise ValueError(f"Input dim mismatch. Expected {self.input_dim}, got {x_flattened_modified.shape[1]}")

        layer_goodness = []
        z = x_flattened_modified.reshape(x_flattened_modified.shape[0], -1)
        z = self._layer_norm(z) # Normalize input

        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            z_act = self.act_fn_eval(z_pre_act) # Use standard ReLU for eval goodness
            goodness = torch.sum(z_act.pow(2), dim=1)
            layer_goodness.append(goodness)
            # Normalize detached activation for next layer's input
            z = self._layer_norm(z_act.detach())

        if len(layer_goodness) != len(self.hidden_dims):
            logger.warning(f"Evaluation Goodness: Scores length ({len(layer_goodness)}) mismatch with num hidden layers ({len(self.hidden_dims)}).")

        return layer_goodness