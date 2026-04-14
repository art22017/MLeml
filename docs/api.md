# API

## `mleml.eml`

```python
from mleml import eml
```

Evaluate the EML operator:

```python
eml(x, y) = exp(x) - log(y)
```

Behavior:

- accepts scalars and array-like inputs
- evaluates using complex arithmetic
- returns real outputs when the imaginary part is numerically negligible

## `mleml.predict`

```python
from mleml import predict
```

Fit a discrete EML expression to point samples.

### Signature

```python
predict(X, Y, max_depth)
```

### Supported inputs

One feature:

```python
predict(X, Y, max_depth=2)
```

Two features:

```python
predict((X1, X2), Y, max_depth=2)
```

Constraints:

- `Y` must be one-dimensional
- all feature arrays must have the same length as `Y`
- current public scope is one or two features

### Return value

`predict` returns a `PredictResult`.

## `PredictResult`

Properties:

- `mse`: training MSE of the snapped expression
- `depth`: effective symbolic depth
- `n_features`: `1` or `2`

Methods:

- `str(result)` returns a readable EML formula
- `result(x)` evaluates a 1D formula
- `result(x1, x2)` evaluates a 2D formula

Example:

```python
import numpy as np
from mleml import predict

x = np.linspace(0.8, 2.0, 64)
y = np.exp(1.0) - np.log(x)

result = predict(x, y, max_depth=1)
print(str(result))
print(result.mse)
print(result(x[:3]))
```

