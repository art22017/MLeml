"""Public package interface for MLeml."""

from .core import eml
from .predictor import PredictResult, predict

__all__ = ["PredictResult", "eml", "predict"]

