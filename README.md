# MLeml

MLeml is a small Python package built around the `EML` operator introduced in the paper *All elementary functions from a single binary operator*.

The operator is:

```python
eml(x, y) = exp(x) - log(y)
```

The package exposes two public entry points:

```python
from mleml import eml, predict
```

`eml` evaluates the primitive operator directly. `predict` fits a shallow EML tree to numerical data and returns the discovered symbolic expression in textual form, for example:

```python
eml(1, eml(x, 1))
```

## Why this package exists

The paper argues that elementary-function expressions can be represented as trees built from one binary operator plus the constant `1`. That gives a uniform grammar:

```text
S -> 1 | x | x1 | x2 | eml(S, S)
```

This package implements a practical subset of that idea for symbolic regression:

- exact `eml` evaluation for scalars and arrays
- a trainable EML tree based on PyTorch
- deterministic multi-restart optimization with `Adam`
- hardening and snapping from soft gates to a discrete formula
- readable string output through `str(result)`

## Installation

### From PyPI

```bash
pip install mleml
```

### From source

```bash
git clone https://github.com/<your-user>/MLeml.git
cd MLeml
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you specifically want a CPU-only local PyTorch install:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Quick start

### Evaluate the EML primitive

```python
from mleml import eml

print(eml(2.0, 3.0))
```

`eml` accepts scalars and array-like inputs. Internally it evaluates in `complex128` and returns real values when the imaginary part is numerically negligible.

### Recover a univariate formula from points

```python
import numpy as np
from mleml import predict

x = np.linspace(0.8, 2.0, 64)
y = np.exp(1.0) - np.log(x)

result = predict(x, y, max_depth=1)

print(result)          # eml(1, x)
print(result.mse)      # close to zero
print(result(x[:5]))   # evaluate the snapped expression
```

### Recover a bivariate formula from points

```python
import numpy as np
from mleml import predict

x1 = np.linspace(0.7, 1.6, 16)
x2 = np.linspace(1.1, 2.0, 16)
y = np.exp(x1) - np.log(x2)

result = predict((x1, x2), y, max_depth=1)

print(result)          # eml(x1, x2)
print(result(x1, x2))
```

## API overview

### `eml(x, y)`

- evaluates `exp(x) - log(y)`
- accepts Python scalars, NumPy arrays, and array-like objects
- uses complex arithmetic internally to preserve the EML semantics

### `predict(X, Y, max_depth)`

- supports one feature: `predict(X, Y, max_depth=...)`
- supports two features: `predict((X1, X2), Y, max_depth=...)`
- returns a `PredictResult`

### `PredictResult`

- `str(result)` returns the snapped EML expression
- `result(x)` evaluates a 1D expression
- `result(x1, x2)` evaluates a 2D expression
- `result.mse` is the training MSE of the snapped tree
- `result.depth` is the effective symbolic depth after snapping
- `result.n_features` is `1` or `2`

## How `predict` works

The model is a full binary EML tree of depth `max_depth`.

- leaves choose among `1`, `x`, `x1`, and `x2` through softmax logits
- each internal node chooses, independently for left and right inputs, whether to pass through the child value or replace it with the constant `1`
- after gating, the node always applies `eml(left, right)`

Optimization uses:

- deterministic multi-restart initialization
- full-batch `Adam`
- temperature annealing during hardening
- penalties for diffuse leaves and non-binary gates
- snapping to a discrete symbolic tree after optimization

The returned formula is always the best snapped candidate found. The function does not require exact recovery to return a result.

## Limitations

- This is a shallow-tree symbolic regression package, not a full theorem prover.
- Exact recovery is realistic mainly for shallow expressions that are naturally expressible as small EML trees.
- For noisy data or functions such as `sin(x)` or `x**8`, the package will usually return a best-fit EML expression rather than an algebraically exact identity.
- The repository examples include both a recoverable EML target and explicit stress tests, so visual output should be interpreted accordingly.
- Internal complex arithmetic and repeated exponentials can cause difficult optimization landscapes for larger depths.
- Runtime increases quickly with depth.

## Repository layout

```text
src/mleml/       package source
tests/           tests and example plots
docs/            API, method, examples, release notes
2603.21852v2.pdf local copy of the reference paper
```

## Development

Create a local environment and install development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the fast test suite:

```bash
pytest -s -m "not slow"
```

Run the example tests that also save plots:

```bash
pytest -s -m slow
```

Build the package:

```bash
python -m build
python -m twine check dist/*
```

## Publishing to PyPI

Short version:

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

For a complete release checklist, see [docs/release.md](docs/release.md).

## GitHub Actions publishing

The repository also supports credential-free PyPI publishing through GitHub Actions Trusted Publishers.

- normal CI runs on `main` and `release`
- PyPI publishing runs only on pushes to the `release` branch
- the publishing workflow file is `.github/workflows/publish.yml`
- the recommended GitHub environment name is `pypi`

See [docs/trusted-publisher.md](docs/trusted-publisher.md) for the exact PyPI pending publisher values.

## Reference

The package is inspired by:

- Andrzej Odrzywolek, *All elementary functions from a single binary operator*, arXiv:2603.21852v2

This repository also keeps the local paper copy in the root directory as `2603.21852v2.pdf`.
