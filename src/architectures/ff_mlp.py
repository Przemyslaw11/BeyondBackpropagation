"""Implements the FF_MLP model for Hinton's Forward-Forward algorithm."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd.function import FunctionCtx

from src.utils.metrics import calculate_accuracy

logger = logging.getLogger(__name__)


class ReLU_full_grad(torch.autograd.Function):
    """A ReLU function that passes the gradient through, ignoring the input value."""

    @staticmethod
    def forward(ctx: FunctionCtx, input_val: torch.Tensor) -> torch.Tensor:
        """Applies the forward pass (standard ReLU)."""
        return input_val.clamp(min=0)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor) -> torch.Tensor:
        """Applies the backward pass (identity function for the gradient)."""
        return grad_output.clone()


class FF_MLP(torch.nn.Module):
    """An MLP for Hinton's Forward-Forward (FF) algorithm.

    This class implements the reference logic, using simultaneous local gradient
    updates via `detach()` and a separate downstream classifier. The structure is
    adapted for the training loop in `src/algorithms/ff.py`.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        input_dim: Optional[int] = None,
        hidden_dims: Optional[List[int]] = None,
        num_classes: Optional[int] = None,
        activation: Optional[str] = "relu",
        bias: Optional[bool] = True,
        norm_eps: Optional[float] = 1.0e-8,
        bias_init: Optional[float] = 0.0,
    ) -> None:
        """Initializes the FF_MLP model.

        Args:
            config: The main experiment configuration dictionary.
            device: The torch.device to run the model on.
            input_dim: Dimensionality of the input features.
            hidden_dims: List of sizes for each hidden layer.
            num_classes: Number of output classes.
            activation: Activation function to use ('relu' or 'tanh').
            bias: Whether to use a bias term in linear layers.
            norm_eps: Epsilon for normalization to prevent division by zero.
            bias_init: Initial value for the bias terms.
        """
        super().__init__()

        self.config = config
        self.model_config = config.get("model", {})
        self.model_params = self.model_config.get("params", {})
        self.algo_params = config.get("algorithm_params", {})
        self.data_config = config.get("data", {})
        self.loader_config = config.get("data_loader", {})
        self.device = device

        if input_dim is None:
            input_channels = self.data_config.get("input_channels", 1)
            image_size = self.data_config.get("image_size", 28)
            self.input_dim = input_channels * image_size * image_size
        else:
            self.input_dim = input_dim

        h_dims_fallback = self.model_params.get("hidden_dims", [1000, 1000, 1000])
        self.hidden_dims = hidden_dims if hidden_dims is not None else h_dims_fallback
        n_classes_fallback = self.data_config.get("num_classes", 10)
        self.num_classes = (
            num_classes if num_classes is not None else n_classes_fallback
        )
        self.num_layers = len(self.hidden_dims)
        act_fallback = self.model_params.get("activation", "ReLU")
        self.activation_name = (
            activation if activation is not None else act_fallback
        ).lower()
        self.use_bias = (
            bias if bias is not None else self.model_params.get("bias", True)
        )
        b_init_fallback = self.model_params.get("bias_init", 0.0)
        self.bias_init = bias_init if bias_init is not None else b_init_fallback
        n_eps_fallback = self.model_params.get("norm_eps", 1.0e-8)
        self.norm_eps = norm_eps if norm_eps is not None else n_eps_fallback
        peer_norm_fb = self.algo_params.get("peer_normalization_factor", 0.0)
        self.peer_normalization_factor = float(peer_norm_fb)
        self.use_peer_normalization = self.peer_normalization_factor > 0.0
        self.peer_momentum = float(self.algo_params.get("peer_momentum", 0.9))

        if self.activation_name == "relu":
            self.act_fn_train = ReLU_full_grad()
            self.act_fn_eval = nn.ReLU()
            if self.bias_init != 0.0:
                logger.info(
                    "%s: Using ReLU activation, forcing bias_init to 0.0 (was %s).",
                    self.__class__.__name__,
                    self.bias_init,
                )
                self.bias_init = 0.0
        elif self.activation_name == "tanh":
            logger.warning("Tanh activation specified, reference uses ReLU.")
            self.act_fn_train = nn.Tanh()
            self.act_fn_eval = nn.Tanh()
            if self.bias_init != 0.0:
                logger.info(
                    "%s: Using Tanh activation, forcing bias_init to 0.0 (was %s).",
                    self.__class__.__name__,
                    self.bias_init,
                )
                self.bias_init = 0.0
        else:
            logger.error(
                "%s: Unsupported activation '%s'. Defaulting to ReLU.",
                self.__class__.__name__,
                self.activation_name,
            )
            self.act_fn_train = ReLU_full_grad()
            self.act_fn_eval = nn.ReLU()
            self.bias_init = 0.0

        # --- FF Layers (nn.Linear) ---
        self.layers = nn.ModuleList()
        current_dim = self.input_dim
        for _, h_dim in enumerate(self.hidden_dims):
            linear_layer = nn.Linear(current_dim, h_dim, bias=self.use_bias)
            self._init_layer_weights(linear_layer)
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
            logger.info(
                "%s: Using Peer Normalization (Factor: %s, Momentum: %s).",
                self.__class__.__name__,
                self.peer_normalization_factor,
                self.peer_momentum,
            )
        else:
            logger.info("%s: Peer Normalization disabled.", self.__class__.__name__)

        # --- Downstream Classification Head ---
        if self.num_layers <= 1:
            logger.warning(
                "%s: <=1 hidden layer. Downstream classifier uses layer 0 activation.",
                self.__class__.__name__,
            )
            channels_for_classification_loss = (
                self.hidden_dims[0] if self.num_layers > 0 else 0
            )
        else:
            # Sum dims from layer 1 (idx=1) up to layer N-1 (idx=N-1)
            channels_for_classification_loss = sum(self.hidden_dims[1:])

        if channels_for_classification_loss <= 0 and self.num_layers > 0:
            logger.error(
                "%s: Downstream classifier input dim is <= 0 (%d). Check config.",
                self.__class__.__name__,
                channels_for_classification_loss,
            )
            channels_for_classification_loss = (
                self.hidden_dims[0] if self.num_layers > 0 else 0
            )
            if channels_for_classification_loss <= 0:
                raise ValueError("Cannot determine downstream classifier input dim.")

        self.linear_classifier = nn.Sequential(
            nn.Linear(
                channels_for_classification_loss, self.num_classes, bias=False
            )  # No bias in ref classifier
        )
        self._init_classifier_weights()

        logger.info(
            (
                "Initialized Modified FF_MLP with %d hidden layers. "
                "Input: %d, Hidden: %s, Classes: %d. "
                "Activation: %s, Bias Init: %s. Peer Norm: %s. Norm Eps: %s. "
                "Downstream classifier input dim: %d."
            ),
            self.num_layers,
            self.input_dim,
            self.hidden_dims,
            self.num_classes,
            self.activation_name,
            self.bias_init,
            self.use_peer_normalization,
            self.norm_eps,
            channels_for_classification_loss,
        )

    def _init_layer_weights(self, layer: nn.Linear) -> None:
        """Initializes weights for a single FF linear layer according to reference."""
        std_dev = 1.0 / math.sqrt(layer.weight.shape[0])
        nn.init.normal_(layer.weight, mean=0, std=std_dev)
        if self.use_bias and layer.bias is not None:
            nn.init.constant_(layer.bias, self.bias_init)

    def _init_classifier_weights(self) -> None:
        """Initializes weights for the downstream linear classifier."""
        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z: torch.Tensor) -> torch.Tensor:
        """Applies RMS length normalization matching reference."""
        rms_norm = torch.sqrt(torch.mean(z**2, dim=-1, keepdim=True))
        return z / (rms_norm + self.norm_eps)

    def _calc_peer_normalization_loss(
        self, layer_idx: int, z_pos: torch.Tensor
    ) -> torch.Tensor:
        """Calculates peer normalization loss matching reference."""
        if not self.use_peer_normalization or layer_idx >= len(self.running_means):
            return torch.zeros(1, device=self.device)
        mean_activity = torch.mean(z_pos, dim=0)
        running_mean = self.running_means[layer_idx]
        new_running_mean = (
            running_mean.detach() * self.peer_momentum
            + mean_activity * (1 - self.peer_momentum)
        )
        self.running_means[layer_idx] = new_running_mean
        peer_loss = (torch.mean(new_running_mean) - new_running_mean) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(
        self, z_pre_norm: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Calculates FF loss and accuracy using a dynamic threshold."""
        sum_of_squares = torch.sum(z_pre_norm**2, dim=-1)
        dynamic_threshold = float(z_pre_norm.shape[1])  # Threshold = number of neurons
        logits = sum_of_squares - dynamic_threshold
        ff_loss = self.ff_loss_criterion(logits, labels.float())
        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels) / z_pre_norm.shape[0]
            ).item() * 100.0
        return ff_loss, ff_accuracy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass using standard activation and normalization."""
        if x.shape[1] != self.input_dim:
            try:
                x = x.view(x.shape[0], self.input_dim)
            except RuntimeError as e:
                msg = f"Cannot reshape input {x.shape} to ({x.shape[0]}, {self.input_dim})"
                raise ValueError(msg) from e
        z = self._layer_norm(x)  # Normalize input
        for _, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            z_act = self.act_fn_eval(z_pre_act)
            z = self._layer_norm(z_act)
        return z

    def forward_ff_train(
        self,
        z_stacked: torch.Tensor,
        posneg_labels: torch.Tensor,
        current_batch_size: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Performs the forward pass for FF training, calculating local losses."""
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.device),
            "Peer_Normalization_Loss_Total": torch.zeros(1, device=self.device),
            "FF_Loss_Total": torch.zeros(1, device=self.device),
        }
        z = z_stacked.reshape(z_stacked.shape[0], -1)
        z = self._layer_norm(z)

        normalized_activations_for_downstream = []

        for idx, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            z_act = self.act_fn_train.apply(z_pre_act)

            # --- Peer Normalization ---
            if self.use_peer_normalization:
                z_pos = z_act[:current_batch_size]
                peer_loss = self._calc_peer_normalization_loss(idx, z_pos)
                scalar_outputs["Peer_Normalization_Loss_Total"] += peer_loss
                # Log per-layer loss
                scalar_outputs[f"Layer_{idx + 1}/Peer_Norm_Loss"] = peer_loss.item()
                scalar_outputs["Loss"] += self.peer_normalization_factor * peer_loss

            # --- Forward-Forward Loss ---
            ff_loss, ff_accuracy = self._calc_ff_loss(z_act, posneg_labels)
            scalar_outputs[f"Layer_{idx + 1}/FF_Loss"] = ff_loss.item()
            scalar_outputs[f"Layer_{idx + 1}/FF_Accuracy"] = ff_accuracy
            scalar_outputs["FF_Loss_Total"] += ff_loss
            scalar_outputs["Loss"] += ff_loss

            # --- Prepare Input for Next Layer ---
            z_detached = z_act.detach()
            z_norm = self._layer_norm(z_detached)

            if idx >= 1:
                normalized_activations_for_downstream.append(
                    z_norm[:current_batch_size].detach()
                )
            z = z_norm

        # --- Assemble Downstream Classifier Input ---
        if normalized_activations_for_downstream:
            try:
                input_classification_model = torch.cat(
                    normalized_activations_for_downstream, dim=-1
                )
            except Exception as e_cat:
                logger.error("FF train cat error: %s", e_cat)
                input_classification_model = torch.zeros(
                    (current_batch_size, 0), device=self.device
                )
        elif self.num_layers == 1:
            logger.warning("FF_MLP has only 1 hidden layer. Downstream input is empty.")
            input_classification_model = torch.zeros(
                (current_batch_size, 0), device=self.device
            )
        else:
            input_classification_model = torch.zeros(
                (current_batch_size, 0), device=self.device
            )

        self._current_downstream_input = input_classification_model
        if any(p.requires_grad for p in self.linear_classifier.parameters()):
            self._current_downstream_input.requires_grad_(True)

        return scalar_outputs["Loss"], scalar_outputs

    def forward_downstream_only(
        self, class_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """Performs the forward pass for the downstream classification model ONLY."""
        if not hasattr(self, "_current_downstream_input"):
            logger.error("Downstream forward called before main FF forward pass.")
            return torch.zeros(1, device=self.device, requires_grad=True), 0.0

        input_cls = self._current_downstream_input
        expected_dim = 0
        try:
            expected_dim = self.linear_classifier[0].in_features
        except (IndexError, AttributeError):
            logger.error("Could not get downstream classifier input dimension.")
            return torch.zeros(1, device=self.device, requires_grad=True), 0.0

        if expected_dim == 0:
            logger.debug("Downstream classifier expects 0 features; zero loss/acc.")
            dummy_loss = torch.zeros(1, device=self.device)
            if any(p.requires_grad for p in self.linear_classifier.parameters()):
                dummy_loss.requires_grad_(True)
            return dummy_loss, 0.0

        if input_cls.shape[1] != expected_dim:
            logger.error(
                "Downstream dim mismatch: Input has %d, classifier expects %d.",
                input_cls.shape[1],
                expected_dim,
            )
            dummy_loss = torch.zeros(1, device=self.device)
            if any(p.requires_grad for p in self.linear_classifier.parameters()):
                dummy_loss.requires_grad_(True)
            return dummy_loss, 0.0

        output = self.linear_classifier(input_cls)
        classification_loss = self.classification_loss_criterion(output, class_labels)
        with torch.no_grad():
            classification_accuracy = calculate_accuracy(output.data, class_labels)

        if hasattr(self, "_current_downstream_input"):
            del self._current_downstream_input

        return classification_loss, classification_accuracy

    def forward_goodness_per_layer(
        self, x_flattened_modified: torch.Tensor
    ) -> List[torch.Tensor]:
        """Forward pass calculating goodness per layer for evaluation."""
        if x_flattened_modified.shape[1] != self.input_dim:
            msg = f"Eval input dim mismatch: {x_flattened_modified.shape[1]} vs {self.input_dim}"
            raise ValueError(msg)

        layer_goodness = []
        z = x_flattened_modified.reshape(x_flattened_modified.shape[0], -1)
        z = self._layer_norm(z)
        for _, layer in enumerate(self.layers):
            z_pre_act = layer(z)
            z_act = self.act_fn_eval(z_pre_act)
            goodness = torch.sum(z_act.pow(2), dim=1)
            layer_goodness.append(goodness)
            z_detached = z_act.detach()
            z = self._layer_norm(z_detached)

        if len(layer_goodness) != len(self.hidden_dims):
            logger.warning(
                "Eval goodness length mismatch: %d vs %d.",
                len(layer_goodness),
                len(self.hidden_dims),
            )
        return layer_goodness
