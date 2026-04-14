from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from mleml import predict


_GENERATED_DIR = Path(__file__).resolve().parent / "generated"


def _plot_fit(path, x, y, x_line, y_line, title: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#f7f1e8")
    ax.set_facecolor("#fffaf4")
    ax.scatter(
        x,
        y,
        s=26,
        alpha=0.78,
        color="#275dad",
        edgecolors="white",
        linewidths=0.4,
        label="samples",
        zorder=3,
    )
    y_line = np.real(np.asarray(y_line))
    ax.plot(x_line, y_line, linewidth=2.6, color="#ca4328", label="EML fit", zorder=4)
    ax.fill_between(x_line, y_line, color="#ca4328", alpha=0.08, zorder=2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.18, color="#715d48")
    for spine in ax.spines.values():
        spine.set_alpha(0.22)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


@pytest.mark.slow
def test_noisy_sine_example(capsys):
    rng = np.random.default_rng(7)
    x = np.linspace(0.7, 2.4, 90)
    y = np.sin(x) + rng.normal(scale=0.03, size=x.shape)
    result = predict(x, y, max_depth=3)

    print(f"sine_result={result}")
    captured = capsys.readouterr()
    assert "sine_result=eml(" in captured.out

    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = result(x_line)
    plot_path = _GENERATED_DIR / "noisy_sine_fit.png"
    _plot_fit(plot_path, x, y, x_line, y_line, "Noisy sin(x) fit with MLeml")

    assert plot_path.exists()
    assert np.isfinite(result.mse)


@pytest.mark.slow
def test_noisy_x8_example(capsys):
    rng = np.random.default_rng(19)
    x = np.linspace(0.65, 1.35, 90)
    y = x**8 + rng.normal(scale=0.03, size=x.shape)
    result = predict(x, y, max_depth=3)

    print(f"x8_result={result}")
    captured = capsys.readouterr()
    assert "x8_result=eml(" in captured.out

    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = result(x_line)
    plot_path = _GENERATED_DIR / "noisy_x8_fit.png"
    _plot_fit(plot_path, x, y, x_line, y_line, "Noisy x^8 fit with MLeml")

    assert plot_path.exists()
    assert np.isfinite(result.mse)
