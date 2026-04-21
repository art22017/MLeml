"""EML tree-based symbolic regression."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from .tree import EMLNode, ONE, Expr, Variable, evaluate_expression, expression_depth


REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128
_EML_CLAMP_DEFAULT = 1.0e300
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


def _safe_eml(left: torch.Tensor, right: torch.Tensor, clamp: float) -> torch.Tensor:
    current = torch.exp(left) - torch.log(right)
    current = torch.complex(
        torch.nan_to_num(current.real, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp),
        torch.nan_to_num(current.imag, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp),
    )
    return current


class _AuthorStyleEMLTree(nn.Module):
    """Author-style EML tree, adapted from v16_final to 1D/2D public API."""

    def __init__(
        self,
        depth: int,
        n_features: int,
        init_scale: float,
        init_strategy: str,
        seed: int,
        eml_clamp: float,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.depth = depth
        self.n_features = n_features
        self.n_terminals = 1 + n_features
        self.n_leaves = 2**depth
        self.n_internal = self.n_leaves - 1
        self.eml_clamp = eml_clamp

        if init_strategy == "biased":
            leaf_init = torch.randn(self.n_leaves, self.n_terminals, dtype=REAL_DTYPE) * init_scale
            leaf_init[:, 0] += 2.0
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
        elif init_strategy == "uniform":
            leaf_init = torch.randn(self.n_leaves, self.n_terminals, dtype=REAL_DTYPE) * init_scale
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
        elif init_strategy == "feature_biased":
            leaf_init = torch.randn(self.n_leaves, self.n_terminals, dtype=REAL_DTYPE) * init_scale
            leaf_init[:, 1:] += 1.0
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 4.0
        elif init_strategy == "random_hot":
            leaf_init = torch.randn(self.n_leaves, self.n_terminals, dtype=REAL_DTYPE) * init_scale
            hot_idx = torch.randint(0, self.n_terminals, (self.n_leaves,))
            leaf_init[torch.arange(self.n_leaves), hot_idx] += 3.0
            gate_init = torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 3.0
            open_mask = torch.rand(self.n_internal, 2) < 0.25
            gate_init[open_mask] -= 6.0
        else:
            raise ValueError(f"Unsupported init strategy: {init_strategy}")

        self.leaf_logits = nn.Parameter(leaf_init)
        self.blend_logits = nn.Parameter(gate_init)

    def forward(
        self,
        features: tuple[torch.Tensor, ...],
        tau_leaf: float,
        tau_gate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        batch_size = features[0].shape[0]
        ones = torch.ones(batch_size, dtype=COMPLEX_DTYPE)
        candidates = torch.stack([ones, *(feature.to(COMPLEX_DTYPE) for feature in features)], dim=1)

        leaf_probs = torch.softmax(self.leaf_logits / tau_leaf, dim=1)
        current_level = torch.matmul(candidates, leaf_probs.to(COMPLEX_DTYPE).T)

        gate_probs_levels = []
        eml_outputs = []
        node_idx = 0

        while current_level.shape[1] > 1:
            n_pairs = current_level.shape[1] // 2
            left_children = current_level[:, 0::2]
            right_children = current_level[:, 1::2]
            gates = torch.sigmoid(self.blend_logits[node_idx : node_idx + n_pairs] / tau_gate)
            gate_probs_levels.append(gates)

            s_left = gates[:, 0].unsqueeze(0)
            s_right = gates[:, 1].unsqueeze(0)
            bypass_left = s_left > _BYPASS_THRESHOLD
            bypass_right = s_right > _BYPASS_THRESHOLD
            one_minus_left = 1.0 - s_left
            one_minus_right = 1.0 - s_right

            left_real = torch.where(bypass_left, 1.0, s_left + one_minus_left * left_children.real)
            left_imag = torch.where(bypass_left, 0.0, one_minus_left * left_children.imag)
            right_real = torch.where(bypass_right, 1.0, s_right + one_minus_right * right_children.real)
            right_imag = torch.where(bypass_right, 0.0, one_minus_right * right_children.imag)

            left_input = torch.complex(left_real, left_imag)
            right_input = torch.complex(right_real, right_imag)
            current_level = _safe_eml(left_input, right_input, self.eml_clamp)
            eml_outputs.append(current_level)
            node_idx += n_pairs

        gate_probs = torch.cat(gate_probs_levels, dim=0)
        return current_level.squeeze(1), leaf_probs, gate_probs, eml_outputs


def _compute_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    leaf_probs: torch.Tensor,
    gate_probs: torch.Tensor,
    eml_outputs: Sequence[torch.Tensor],
    lam_ent: float,
    lam_bin: float,
    lam_inter: float,
    inter_threshold: float,
    uncertainty_power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data_loss = torch.mean((pred - target).abs() ** 2).real
    eps = 1e-12

    if leaf_probs.shape[1] > 1:
        leaf_max = leaf_probs.max(dim=1).values
        norm = 1.0 - 1.0 / leaf_probs.shape[1]
        leaf_unc = torch.clamp((1.0 - leaf_max) / norm, 0.0, 1.0).pow(uncertainty_power)
    else:
        leaf_unc = torch.zeros(leaf_probs.shape[0], dtype=REAL_DTYPE)
    leaf_ent = -(leaf_probs * (leaf_probs + eps).log()).sum(dim=1)
    entropy = (leaf_ent * leaf_unc).mean()

    gate_unc = torch.clamp(1.0 - (2.0 * gate_probs - 1.0).abs(), 0.0, 1.0).pow(uncertainty_power)
    gate_bin = gate_probs * (1.0 - gate_probs)
    binarity = (gate_bin * gate_unc).mean()

    inter_penalty = torch.tensor(0.0, dtype=REAL_DTYPE)
    if lam_inter > 0 and eml_outputs:
        penalties = []
        for level_output in eml_outputs:
            excess = torch.relu(level_output.abs() - inter_threshold)
            penalties.append(excess.pow(2).mean().real)
        inter_penalty = torch.stack(penalties).mean()

    total = data_loss + lam_ent * entropy + lam_bin * binarity + lam_inter * inter_penalty
    return total, data_loss, entropy, binarity, inter_penalty


def _evaluate_soft(
    tree: _AuthorStyleEMLTree,
    features: tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    tau: float,
) -> float:
    with torch.no_grad():
        pred, _, _, _ = tree(features, tau_leaf=tau, tau_gate=tau)
        return float(torch.mean((pred - targets).abs() ** 2).real.item())


def _snap_expression(tree: _AuthorStyleEMLTree, variable_names: tuple[str, ...]) -> Expr:
    leaf_choice = torch.argmax(tree.leaf_logits.detach(), dim=1).cpu().tolist()
    gate_choice = (tree.blend_logits.detach() >= 0).cpu().tolist()
    terminals = [ONE, *(Variable(name, index) for index, name in enumerate(variable_names))]

    flat_idx = 0
    levels: list[list[Expr]] = []
    for width in (2**level for level in range(tree.depth, -1, -1)):
        if width == tree.n_leaves:
            levels.append([terminals[idx] for idx in leaf_choice])
            continue

        child_level = levels[-1]
        current = []
        for pos in range(width):
            gate = gate_choice[flat_idx]
            left_expr = ONE if gate[0] else child_level[2 * pos]
            right_expr = ONE if gate[1] else child_level[2 * pos + 1]
            current.append(EMLNode(left_expr, right_expr))
            flat_idx += 1
        levels.append(current)

    return levels[-1][0]


def _project_hard(tree: _AuthorStyleEMLTree, k: float = 24.0) -> _AuthorStyleEMLTree:
    snapped = deepcopy(tree)
    with torch.no_grad():
        leaf_choice = torch.argmax(snapped.leaf_logits, dim=1)
        new_leaf = torch.full_like(snapped.leaf_logits, -k)
        new_leaf[torch.arange(snapped.n_leaves), leaf_choice] = k
        snapped.leaf_logits.copy_(new_leaf)

        gate_binary = (snapped.blend_logits >= 0).to(snapped.blend_logits.dtype)
        new_gate = torch.where(
            gate_binary > 0.5,
            torch.full_like(snapped.blend_logits, k),
            torch.full_like(snapped.blend_logits, -k),
        )
        snapped.blend_logits.copy_(new_gate)
    return snapped


def _enumerate_shallow_candidates(variable_names: tuple[str, ...]) -> list[Expr]:
    terminals = [ONE, *(Variable(name, index) for index, name in enumerate(variable_names))]
    expressions: list[Expr] = []
    expressions.extend(terminals)
    for left in terminals:
        for right in terminals:
            expressions.append(EMLNode(left, right))
    return expressions


def _best_enumerated_candidate(
    features: tuple[np.ndarray, ...],
    target: np.ndarray,
    variable_names: tuple[str, ...],
) -> _Candidate:
    best_expr = min(
        _enumerate_shallow_candidates(variable_names),
        key=lambda expr: _mean_squared_error(evaluate_expression(expr, features), target),
    )
    return _Candidate(
        expression=best_expr,
        mse=_mean_squared_error(evaluate_expression(best_expr, features), target),
        depth=expression_depth(best_expr),
    )


def _train_once(
    features: tuple[np.ndarray, ...],
    target: np.ndarray,
    depth: int,
    variable_names: tuple[str, ...],
    init_strategy: str,
    seed: int,
) -> _Candidate:
    feature_tensors = tuple(torch.tensor(feature, dtype=REAL_DTYPE) for feature in features)
    target_tensor = torch.tensor(target, dtype=COMPLEX_DTYPE)

    tree = _AuthorStyleEMLTree(
        depth=depth,
        n_features=len(features),
        init_scale=1.0,
        init_strategy=init_strategy,
        seed=seed,
        eml_clamp=_EML_CLAMP_DEFAULT,
    )
    optimizer = torch.optim.Adam(tree.parameters(), lr=0.01)

    search_iters = 120 + 40 * depth
    hardening_iters = 60 + 20 * depth
    tau_search = 2.5
    tau_hard = 0.01
    lam_inter = 1e-4
    inter_threshold = 50.0

    phase = "search"
    hard_step = 0
    patience = max(40, search_iters // 2)
    patience_threshold = 1e-2
    plateau_rtol = 1e-3
    tail_eval_tau = 0.2
    best_soft_loss = float("inf")
    best_soft_state: dict[str, torch.Tensor] | None = None
    best_hard_loss = float("inf")
    best_hard_state: dict[str, torch.Tensor] | None = None
    plateau_counter = 0
    hard_trigger_streak = 0
    hard_success_streak = 0

    total_iters = search_iters + hardening_iters
    for iteration in range(1, total_iters + 1):
        if phase == "search":
            if iteration > search_iters or (plateau_counter >= patience and best_soft_loss < patience_threshold):
                phase = "hardening"
                hard_step = 0
                if best_soft_state is not None:
                    tree.load_state_dict(best_soft_state)
                    optimizer = torch.optim.Adam(tree.parameters(), lr=0.01)

        if phase == "search":
            tau = tau_search
            lam_ent = 0.0
            lam_bin = 0.0
            lr_mult = 1.0
        else:
            if hard_step >= hardening_iters:
                break
            t = hard_step / max(1, hardening_iters)
            t_tau = t**2.0
            tau = tau_search * (tau_hard / tau_search) ** t_tau
            lam_ent = t * 2e-2
            lam_bin = t * 2e-2
            lr_mult = max(0.01, (1.0 - t) ** 2)
            hard_step += 1

        optimizer.param_groups[0]["lr"] = 0.01 * lr_mult
        optimizer.zero_grad()

        pred, leaf_probs, gate_probs, eml_outputs = tree(feature_tensors, tau_leaf=tau, tau_gate=tau)
        total, data_loss, _, _, _ = _compute_losses(
            pred,
            target_tensor,
            leaf_probs,
            gate_probs,
            eml_outputs,
            lam_ent=lam_ent,
            lam_bin=lam_bin,
            lam_inter=lam_inter,
            inter_threshold=inter_threshold,
            uncertainty_power=1.0,
        )
        if not torch.isfinite(total):
            continue

        total.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        optimizer.step()

        soft_loss = float(data_loss.item())
        if np.isfinite(soft_loss) and soft_loss < best_soft_loss:
            rel_imp = (best_soft_loss - soft_loss) / max(best_soft_loss, 1e-15)
            best_soft_loss = soft_loss
            best_soft_state = _snapshot(tree)
            plateau_counter = 0 if rel_imp > plateau_rtol else plateau_counter + 1
        else:
            plateau_counter += 1

        do_eval = iteration % 30 == 0
        if phase == "hardening" and tau <= tail_eval_tau:
            do_eval = do_eval or (iteration % 10 == 0)
        if do_eval:
            hard_mse = _evaluate_soft(tree, feature_tensors, target_tensor, tau=tau_hard)
            if np.isfinite(hard_mse) and hard_mse < best_hard_loss:
                best_hard_loss = hard_mse
                best_hard_state = _snapshot(tree)

            if phase == "hardening" and np.isfinite(hard_mse) and hard_mse < 1e-20:
                hard_success_streak += 1
            elif phase == "hardening":
                hard_success_streak = 0

            if phase == "search" and np.isfinite(hard_mse) and hard_mse < 1e-20:
                hard_trigger_streak += 1
            elif phase == "search":
                hard_trigger_streak = 0

            if hard_success_streak >= 3:
                break
            if phase == "search" and hard_trigger_streak >= 3:
                phase = "hardening"
                hard_step = 0
                if best_soft_state is not None:
                    tree.load_state_dict(best_soft_state)
                    optimizer = torch.optim.Adam(tree.parameters(), lr=0.01)

    if best_hard_state is not None:
        tree.load_state_dict(best_hard_state)
    elif best_soft_state is not None:
        tree.load_state_dict(best_soft_state)

    if depth <= 1:
        lbfgs = torch.optim.LBFGS(
            tree.parameters(),
            lr=0.6,
            max_iter=8,
            history_size=30,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            pred, leaf_probs, gate_probs, eml_outputs = tree(feature_tensors, tau_leaf=tau_hard, tau_gate=tau_hard)
            total, _, _, _, _ = _compute_losses(
                pred,
                target_tensor,
                leaf_probs,
                gate_probs,
                eml_outputs,
                lam_ent=2e-2,
                lam_bin=2e-2,
                lam_inter=lam_inter,
                inter_threshold=inter_threshold,
                uncertainty_power=1.0,
            )
            total.backward()
            return total

        try:
            lbfgs.step(closure)
        except Exception:
            pass

    snapped_tree = _project_hard(tree)
    expression = _snap_expression(snapped_tree, variable_names)
    mse = _mean_squared_error(evaluate_expression(expression, features), target)
    return _Candidate(expression=expression, mse=mse, depth=expression_depth(expression))


def predict(X: Any, Y: Any, max_depth: int) -> PredictResult:
    """Fit the best snapped EML expression for one- or two-feature data."""
    if not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be an integer greater than or equal to 1.")

    features, target, variable_names = _coerce_training_data(X, Y)
    if max_depth == 1:
        best = _best_enumerated_candidate(features, target, variable_names)
        return PredictResult(expression=best.expression, mse=best.mse, depth=best.depth, n_features=len(features))

    init_strategies = ["biased", "random_hot"]
    if len(features) == 2:
        init_strategies.insert(1, "feature_biased")
    seeds = [137]
    candidates = [
        _train_once(
            features=features,
            target=target,
            depth=max_depth,
            variable_names=variable_names,
            init_strategy=strategy,
            seed=seed,
        )
        for strategy in init_strategies
        for seed in seeds
    ]
    best = min(candidates, key=lambda candidate: candidate.mse)
    return PredictResult(expression=best.expression, mse=best.mse, depth=best.depth, n_features=len(features))
