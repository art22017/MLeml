"""Expression tree primitives for snapped EML formulas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from .core import eml


@dataclass(frozen=True)
class ConstantOne:
    def __str__(self) -> str:
        return "1"


@dataclass(frozen=True)
class Variable:
    name: str
    index: int

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class EMLNode:
    left: "Expr"
    right: "Expr"

    def __str__(self) -> str:
        return f"eml({self.left}, {self.right})"


Expr = Union[ConstantOne, Variable, EMLNode]


ONE = ConstantOne()


def expression_depth(expr: Expr) -> int:
    if isinstance(expr, (ConstantOne, Variable)):
        return 0
    return 1 + max(expression_depth(expr.left), expression_depth(expr.right))


def evaluate_expression(expr: Expr, features: Tuple[np.ndarray, ...]):
    if isinstance(expr, ConstantOne):
        return np.ones_like(features[0], dtype=np.float64)
    if isinstance(expr, Variable):
        return features[expr.index]
    return eml(evaluate_expression(expr.left, features), evaluate_expression(expr.right, features))

