"""Low-level EML operator utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None


def _finish_numpy(value: np.ndarray | np.generic) -> Any:
    cast = np.real_if_close(value, tol=1000)
    if np.isscalar(cast):
        return cast.item()
    return cast


def eml(x: Any, y: Any) -> Any:
    """Evaluate exp(x) - log(y) using complex arithmetic."""
    if torch is not None and (isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor)):
        x_t = torch.as_tensor(x, dtype=torch.complex128)
        y_t = torch.as_tensor(y, dtype=torch.complex128)
        out = torch.exp(x_t) - torch.log(y_t)
        if torch.max(torch.abs(out.imag)).item() < 1e-9:
            return out.real
        return out

    x_arr = np.asarray(x, dtype=np.complex128)
    y_arr = np.asarray(y, dtype=np.complex128)
    out = np.exp(x_arr) - np.log(y_arr)
    return _finish_numpy(out)

