# Examples

## Exact shallow recovery

Univariate:

```python
import numpy as np
from mleml import predict

x = np.linspace(0.8, 2.0, 64)
y = np.exp(1.0) - np.log(x)

result = predict(x, y, max_depth=1)
print(result)
```

Bivariate:

```python
import numpy as np
from mleml import predict

x1 = np.linspace(0.7, 1.6, 64)
x2 = np.linspace(1.0, 2.0, 64)
y = np.exp(x1) - np.log(x2)

result = predict((x1, x2), y, max_depth=1)
print(result)
```

## Noisy examples

The repository includes slow example tests for:

- a noisy target that is actually representable by a shallow EML tree
- noisy `sin(x)`
- noisy `x**8`

Run them with:

```bash
pytest -m slow -s tests/test_examples.py
```

They generate:

- the discovered EML formula printed to stdout
- a Matplotlib plot overlaying raw points and the snapped EML curve
- stable PNG outputs under `tests/generated/`

Interpretation:

- the recoverable noisy EML target is the positive demo and should fit well
- `sin(x)` and `x**8` are stress tests and are not expected to match closely with the current shallow grammar

These examples are intended as smoke tests and demos, not as exact symbolic benchmarks.
