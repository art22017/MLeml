from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from mleml import predict


def _plot_fit(path, x, y, x_line, y_line, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=18, alpha=0.7, label="samples")
    ax.plot(x_line, np.real(np.asarray(y_line)), linewidth=2.0, label="EML fit")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@pytest.mark.slow
def test_noisy_sine_example(tmp_path, capsys):
    rng = np.random.default_rng(7)
    x = np.linspace(0.7, 2.4, 90)
    y = np.sin(x) + rng.normal(scale=0.03, size=x.shape)
    result = predict(x, y, max_depth=3)

    print(f"sine_result={result}")
    captured = capsys.readouterr()
    assert "sine_result=eml(" in captured.out

    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = result(x_line)
    plot_path = tmp_path / "noisy_sine_fit.png"
    _plot_fit(plot_path, x, y, x_line, y_line, "Noisy sin(x) fit with MLeml")

    assert plot_path.exists()
    assert np.isfinite(result.mse)


@pytest.mark.slow
def test_noisy_x8_example(tmp_path, capsys):
    rng = np.random.default_rng(19)
    x = np.linspace(0.65, 1.35, 90)
    y = x**8 + rng.normal(scale=0.03, size=x.shape)
    result = predict(x, y, max_depth=3)

    print(f"x8_result={result}")
    captured = capsys.readouterr()
    assert "x8_result=eml(" in captured.out

    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = result(x_line)
    plot_path = tmp_path / "noisy_x8_fit.png"
    _plot_fit(plot_path, x, y, x_line, y_line, "Noisy x^8 fit with MLeml")

    assert plot_path.exists()
    assert np.isfinite(result.mse)
