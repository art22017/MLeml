"""EML tree-based symbolic regression."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .tree import EMLNode, ONE, Expr, Variable, evaluate_expression, expression_depth


REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
_BYPASS_THRESHOLD = 1.0 - torch.finfo(torch.float64).eps


@dataclass(frozen=True)
class PredictResult:
    """Final snapped symbolic expression returned by predict()."""

    expression: Expr
    mse: float
    depth: int
    n_features: int

    def __str__(self) -> str:
        return str(self.expression)

    def __call__(self, *args: Any):
        if self.n_features == 1:
            if len(args) != 1:
                raise TypeError("Expected exactly one positional argument for a 1D result.")
            features = (_coerce_eval_feature(args[0]),)
        else:
            if len(args) == 1 and isinstance(args[0], tuple):
                args = args[0]
            if len(args) != 2:
                raise TypeError("Expected exactly two positional arguments for a 2D result.")
            features = (_coerce_eval_feature(args[0]), _coerce_eval_feature(args[1]))
        return evaluate_expression(self.expression, features)


@dataclass(frozen=True)
class _Candidate:
    expression: Expr
    mse: float
    depth: int


def _coerce_eval_feature(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _coerce_training_data(X: Any, Y: Any) -> tuple[tuple[np.ndarray, ...], np.ndarray, tuple[str, ...]]:
    y = np.asarray(Y, dtype=np.float64).reshape(-1)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("Y must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Y must contain only finite values.")

    if isinstance(X, tuple):
        if len(X) != 2:
            raise ValueError("Tuple input is supported only for two feature arrays.")
        features = tuple(np.asarray(feature, dtype=np.float64).reshape(-1) for feature in X)
        names = ("x1", "x2")
    else:
        features = (np.asarray(X, dtype=np.float64).reshape(-1),)
        names = ("x",)

    if any(feature.size != y.size for feature in features):
        raise ValueError("All feature arrays must have the same length as Y.")
    if any(not np.all(np.isfinite(feature)) for feature in features):
        raise ValueError("X must contain only finite values.")

    return features, y, names


def _mean_squared_error(prediction: Any, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.complex128)
    return float(np.mean(np.abs(pred - target) ** 2))


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _node_index(level: int, position: int) -> int:
    return (1 << level) - 1 + position


def _safe_eml(left: torch.Tensor, right: torch.Tensor, clamp: float) -> torch.Tensor:
    value = torch.exp(left) - torch.log(right)
    real = torch.nan_to_num(value.real, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)
    imag = torch.nan_to_num(value.imag, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)
    return torch.complex(real, imag)


def _blend_with_one(child: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    gate = gate.to(REAL_DTYPE)
    bypass = gate > _BYPASS_THRESHOLD
    one_minus = 1.0 - gate
    real = torch.where(bypass, torch.ones_like(child.real), gate + one_minus * child.real)
    imag = torch.where(bypass, torch.zeros_like(child.imag), one_minus * child.imag)
    return torch.complex(real, imag)


class _DifferentiableEMLTree(nn.Module):
    def __init__(
        self,
        depth: int,
        n_features: int,
        init_mode: str,
        seed: int,
        clamp: float = 200.0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.depth = depth
        self.n_features = n_features
        self.n_terminals = 1 + n_features
        self.n_leaves = 1 << depth
        self.n_internal = self.n_leaves - 1
        self.clamp = clamp

        leaf_init = torch.randn(self.n_leaves, self.n_terminals, dtype=REAL_DTYPE) * 0.25
        gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * 0.35

        if init_mode == "constant_bias":
            leaf_init[:, 0] += 2.5
            gate_init += 1.5
        elif init_mode.startswith("feature_bias_"):
            feature_index = int(init_mode.rsplit("_", 1)[1]) + 1
            leaf_init[:, feature_index] += 2.2
            gate_init += 0.8
        elif init_mode == "mixed_hot":
            hot = torch.randint(0, self.n_terminals, (self.n_leaves,))
            leaf_init[torch.arange(self.n_leaves), hot] += 2.5
            open_mask = torch.rand(self.n_internal, 2) > 0.5
            gate_init[open_mask] -= 2.0

        self.leaf_logits = nn.Parameter(leaf_init)
        self.gate_logits = nn.Parameter(gate_init)

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        tau_leaf: float,
        tau_gate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        ones = torch.ones_like(features[0], dtype=COMPLEX_DTYPE)
        candidates = torch.stack([ones, *(feature.to(COMPLEX_DTYPE) for feature in features)], dim=1)
        leaf_probs = torch.softmax(self.leaf_logits / tau_leaf, dim=1)
        leaf_values = torch.matmul(candidates, leaf_probs.transpose(0, 1).to(COMPLEX_DTYPE))

        gate_probs: list[torch.Tensor] = []
        intermediate_outputs: list[torch.Tensor] = []

        def eval_node(level: int, position: int) -> torch.Tensor:
            if level == self.depth:
                return leaf_values[:, position]

            left_child = eval_node(level + 1, 2 * position)
            right_child = eval_node(level + 1, 2 * position + 1)
            gate = torch.sigmoid(self.gate_logits[_node_index(level, position)] / tau_gate)
            gate_probs.append(gate)
            left_input = _blend_with_one(left_child, gate[0])
            right_input = _blend_with_one(right_child, gate[1])
            output = _safe_eml(left_input, right_input, self.clamp)
            intermediate_outputs.append(output)
            return output

        prediction = eval_node(0, 0)
        stacked_gates = torch.stack(gate_probs, dim=0)
        return prediction, leaf_probs, stacked_gates, intermediate_outputs


def _objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    leaf_probs: torch.Tensor,
    gate_probs: torch.Tensor,
    intermediate_outputs: Sequence[torch.Tensor],
    entropy_weight: float,
    binarity_weight: float,
    magnitude_weight: float,
    magnitude_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    data_loss = torch.mean(torch.abs(prediction - target) ** 2).real
    leaf_entropy = -(leaf_probs * torch.log(leaf_probs + 1e-12)).sum(dim=1).mean()
    gate_binarity = (gate_probs * (1.0 - gate_probs)).mean()

    magnitude_penalty = torch.tensor(0.0, dtype=REAL_DTYPE)
    if intermediate_outputs:
        penalties = []
        for output in intermediate_outputs:
            excess = torch.relu(torch.abs(output) - magnitude_threshold)
            penalties.append(torch.mean(excess**2).real)
        magnitude_penalty = torch.stack(penalties).mean()

    total = (
        data_loss
        + entropy_weight * leaf_entropy
        + binarity_weight * gate_binarity
        + magnitude_weight * magnitude_penalty
    )
    return total, data_loss


def _snap_expression(
    tree: _DifferentiableEMLTree,
    variable_names: tuple[str, ...],
) -> Expr:
    leaf_choice = torch.argmax(tree.leaf_logits.detach(), dim=1).cpu().tolist()
    gate_choice = (torch.sigmoid(tree.gate_logits.detach()) >= 0.5).cpu().tolist()
    terminals = [ONE, *(Variable(name, index) for index, name in enumerate(variable_names))]

    def build(level: int, position: int) -> Expr:
        if level == tree.depth:
            return terminals[leaf_choice[position]]

        node_gate = gate_choice[_node_index(level, position)]
        left_expr = ONE if node_gate[0] else build(level + 1, 2 * position)
        right_expr = ONE if node_gate[1] else build(level + 1, 2 * position + 1)
        return EMLNode(left_expr, right_expr)

    return build(0, 0)


def _train_once(
    features: tuple[np.ndarray, ...],
    target: np.ndarray,
    depth: int,
    variable_names: tuple[str, ...],
    init_mode: str,
    seed: int,
) -> _Candidate:
    feature_tensors = tuple(torch.tensor(feature, dtype=REAL_DTYPE) for feature in features)
    target_tensor = torch.tensor(target, dtype=COMPLEX_DTYPE)

    tree = _DifferentiableEMLTree(depth=depth, n_features=len(features), init_mode=init_mode, seed=seed)
    optimizer = torch.optim.Adam(tree.parameters(), lr=0.05 if depth <= 2 else 0.035)

    best_data_loss = float("inf")
    best_state = _snapshot(tree)

    search_steps = 180 + 60 * depth
    hard_steps = 120 + 40 * depth

    for _ in range(search_steps):
        prediction, leaf_probs, gate_probs, intermediate_outputs = tree(feature_tensors, tau_leaf=1.0, tau_gate=1.0)
        total, data_loss = _objective(
            prediction=prediction,
            target=target_tensor,
            leaf_probs=leaf_probs,
            gate_probs=gate_probs,
            intermediate_outputs=intermediate_outputs,
            entropy_weight=0.01,
            binarity_weight=0.01,
            magnitude_weight=1e-4,
            magnitude_threshold=40.0,
        )
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), max_norm=5.0)
        optimizer.step()

        data_value = float(data_loss.item())
        if data_value < best_data_loss:
            best_data_loss = data_value
            best_state = _snapshot(tree)

    optimizer.param_groups[0]["lr"] *= 0.5
    for step in range(hard_steps):
        fraction = step / max(hard_steps - 1, 1)
        tau = max(0.05, 0.9 - 0.85 * fraction)
        prediction, leaf_probs, gate_probs, intermediate_outputs = tree(feature_tensors, tau_leaf=tau, tau_gate=tau)
        total, data_loss = _objective(
            prediction=prediction,
            target=target_tensor,
            leaf_probs=leaf_probs,
            gate_probs=gate_probs,
            intermediate_outputs=intermediate_outputs,
            entropy_weight=0.02 + 0.06 * fraction,
            binarity_weight=0.04 + 0.12 * fraction,
            magnitude_weight=1e-4,
            magnitude_threshold=40.0,
        )
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), max_norm=5.0)
        optimizer.step()

        data_value = float(data_loss.item())
        if data_value < best_data_loss:
            best_data_loss = data_value
            best_state = _snapshot(tree)

    candidate_states = [best_state, _snapshot(tree)]
    best_candidate: _Candidate | None = None

    for state in candidate_states:
        tree.load_state_dict(deepcopy(state))
        expression = _snap_expression(tree, variable_names)
        mse = _mean_squared_error(evaluate_expression(expression, features), target)
        candidate = _Candidate(expression=expression, mse=mse, depth=expression_depth(expression))
        if best_candidate is None or candidate.mse < best_candidate.mse:
            best_candidate = candidate

    assert best_candidate is not None
    return best_candidate


def predict(X: Any, Y: Any, max_depth: int) -> PredictResult:
    """Fit the best snapped EML expression for one- or two-feature data."""
    if not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be an integer greater than or equal to 1.")

    features, target, variable_names = _coerce_training_data(X, Y)
    init_modes = ["constant_bias", "uniform", "mixed_hot"]
    init_modes.extend(f"feature_bias_{index}" for index in range(len(features)))

    seeds = [11, 23, 37, 47, 59, 71, 89]
    candidates = [
        _train_once(
            features=features,
            target=target,
            depth=max_depth,
            variable_names=variable_names,
            init_mode=mode,
            seed=seeds[index],
        )
        for index, mode in enumerate(init_modes)
    ]
    best = min(candidates, key=lambda candidate: candidate.mse)
    return PredictResult(expression=best.expression, mse=best.mse, depth=best.depth, n_features=len(features))

