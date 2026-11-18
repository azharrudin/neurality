# Backends: NumPy and native Python

[Documentation index](README.md) · [Architecture](architecture.md) · [API reference](api.md) ·
[Initialization](initialization.md) · [Serialization](serialization.md)

Neurality runs on one of two interchangeable backends. The model mathematics and the public API
are the same; the difference is how tensors are represented and which library performs the
operations.

| Property | Native Python | NumPy |
|---|---|---|
| Enum | `N.Backends.PYTHON` | `N.Backends.NUMPY` |
| String value | `"python"` | `"numpy"` |
| Tensor representation | Nested Python lists of `float` | `numpy.ndarray` |
| External dependency | Standard library only | NumPy |
| Default backend | No | Yes |
| Random generator | `random` | `numpy.random` |
| Elementwise shape rules | Equal shapes required | NumPy broadcasting allowed |
| Matrix multiplication | Pure-Python loops | NumPy matrix multiplication |
| Typical speed | Slower | Faster, especially for dense layers |
| Portability | Very high | Requires NumPy |

Both backends execute the same forward and backward algorithms. Backend selection does not
change layer parameters, loss definitions, mask semantics, or JSON structure.

## Selecting a backend

Pass the enum or its string value at construction:

```python
from lib.neurality import Neurality as N

native_model = N(backend=N.Backends.PYTHON)
numpy_model = N(backend=N.Backends.NUMPY)

# String values are accepted too.
same_as_native = N(backend="python")
```

`N.backends` is an alias of `N.Backends`:

```python
model = N(backend=N.backends.PYTHON)
```

### The default requires NumPy

```python
model = N()  # NumPy backend
```

Constructing `N()` immediately imports NumPy. On a system without NumPy it raises:

```text
ImportError: NumPy is required for the NUMPY backend; select Backends.PYTHON instead
```

Calling `model.backend(N.Backends.PYTHON)` afterward cannot rescue construction, because the
constructor already failed. Select the native backend explicitly when NumPy may be absent:

```python
try:
    import numpy  # noqa: F401
    backend = N.Backends.NUMPY
except ImportError:
    backend = N.Backends.PYTHON

model = N(backend=backend)
```

### Inspecting the active backend

`model.backend` is a **method** (the setter), not a property. The active enum is stored in the
private attribute `model._backend`:

```python
print(model._backend)  # Backends.PYTHON or Backends.NUMPY
```

Application code should normally use the documented setter instead of reading private state.

## Native Python backend

The native backend uses nested lists:

```python
model = N(backend=N.Backends.PYTHON).fit([[0.0, 1.0]])
model.dense(1, N.Activations.sigmoid)

output = model.forward([[0.0, 1.0]])
print(type(output))        # <class 'list'>
print(type(output[0]))     # <class 'list'>
print(type(output[0][0]))  # <class 'float'>
```

Characteristics:

- **No NumPy import.** The backend uses only the standard library, including `math` and
  `random`.
- **Strict elementwise shapes.** When two operands are lists, they must have the same shape.
  A list can be combined with a scalar, but there is no array broadcasting.
- **Readable internals.** Activations, gradients, and caches are ordinary nested lists, which
  makes the implementation easy to inspect and debug.
- **Higher overhead.** Matrix multiplication and repeated elementwise operations run as Python
  loops, so training is slower than with NumPy.
- **Full feature parity.** Dense layers, embeddings, attention, normalization, residuals,
  losses, optimizers, masks, and JSON persistence all work.

Use the native backend for:

- environments where NumPy is unavailable;
- teaching, debugging, or tracing exact values;
- small models where portability matters more than speed;
- verifying that a model behaves the same without NumPy.

## NumPy backend

The NumPy backend stores parameters and activations as floating-point arrays:

```python
model = N(backend=N.Backends.NUMPY).fit([[0.0, 1.0]])
model.dense(1, N.Activations.sigmoid)

output = model.forward([[0.0, 1.0]])
print(type(output))        # <class 'numpy.ndarray'>
print(type(output[0]))     # <class 'numpy.ndarray'>
print(type(output[0][0]))  # <class 'numpy.float64'>
```

Characteristics:

- **Copied arrays.** `N.Math.array(value, backend)` converts input with
  `numpy.array(value, dtype=float, copy=True)`. Inputs are not aliased into model state.
- **NumPy broadcasting.** Elementwise operations accept broadcast-compatible shapes. This is
  convenient, but it can hide a target-shape error in a loss call. Supply exact shapes.
- **Faster matrix operations.** Dense forward and backward passes use array multiplication
  rather than Python loops.
- **Not fully vectorized.** Several paths still loop in Python: embedding lookups, positional
  addition, per-row normalization, attention over batch items and heads, and some caches.
  Selecting NumPy does not make every layer a single vectorized kernel.
- **No GPU execution.** NumPy runs on the CPU; there is no CUDA or other accelerator path.

Use the NumPy backend for:

- normal training and inference workloads;
- larger batches and wider dense layers;
- experiments that already use NumPy arrays for data preparation.

## Switching backends at runtime

`model.backend(...)` converts the model in place:

```python
import random
from lib.neurality import Neurality as N

random.seed(42)
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

model = N(backend=N.Backends.PYTHON).fit(X)
model.dense(4, N.Activations.tanh, initialization=N.Initializations.glorot)
model.dense(1, N.Activations.sigmoid, initialization=N.Initializations.glorot)

native_output = model.forward(X)

model.backend(N.Backends.NUMPY)
numpy_output = model.forward(X)

difference = max(
    abs(float(a[0]) - float(b[0]))
    for a, b in zip(native_output, numpy_output)
)
print(difference)  # 0.0, or a very small floating-point value
```

The conversion performs these steps:

1. Selects and validates the target enum.
2. Imports NumPy immediately when switching to the NumPy backend.
3. Converts dense neuron weights stored in `model.perceptrons`.
4. Converts sequence-layer parameters: embedding and positional tables, attention projections,
   normalization `gamma` and `beta`.
5. Converts stored `fit()` data and target arrays, if present.
6. Sets the active backend.
7. **Resets optimizer state** (moments, velocities, squared-gradient averages, step counts).
8. **Clears forward caches** (`output`, `layer_inputs`, `layer_linear`, `_caches`).

It returns the model, so it can be chained.

Important consequences:

- Parameters are preserved; training state is not. Adam, RMSprop, and Momentum restart from
  zero after a switch.
- A forward pass must run again before `backward()`.
- Reselecting the same backend still resets optimizer state and clears caches.
- Avoid switching backends inside a training loop. Select the backend once, or switch after
  saving a checkpoint when the optimizer state is not needed.

To continue training after conversion, attach an optimizer again and run a fresh forward:

```python
y = [[0.0], [1.0], [1.0], [1.0]]

model.optimizer(N.Optimizers.Adam(learning_rate=0.01))
prediction = model.forward(X)
model.backward(y, N.Gradients.gradient_bce)
```

## Randomness and initialization

The two backends use different random generators:

| Initialization path | Native Python | NumPy |
|---|---|---|
| `dense()` built-in/default initializers | `random.gauss` | `numpy.random.randn` |
| Embedding, positional, attention layers | `random.gauss` | `numpy.random.randn` |
| Direct `N.Math.randn()` | Global default backend | Global default backend |
| Direct `N.Initializations.*()` | Global default backend | Global default backend |

Consequences:

- Seed `random.seed(...)` before building a native model and `numpy.random.seed(...)` before
  building a NumPy model.
- Matching integer seeds do **not** produce identical weights, because the generators and
  distributions are different.
- `N.Math.randn()` follows the internal global `_default_backend`, not the backend of a
  particular model. Built-in initializers passed to `dense()` are special-cased to use the
  model's backend. A custom initializer chooses its own randomness.

For a cross-backend numerical comparison, build one model and convert or reload it instead of
reinitializing it. See [weight initialization](initialization.md#reproducibility).

## Numerical behavior and parity

Both backends use floating-point numbers and the same formulas. Small differences can still
appear because operation order and library kernels differ. Continuing the conversion example
above, compare the two outputs with a tolerance rather than exact equality:

```python
difference = max(
    abs(float(a) - float(b))
    for row_a, row_b in zip(native_output, numpy_output)
    for a, b in zip(row_a, row_b)
)
assert difference < 1e-9
```

A converted model often produces identical dense outputs in practice, but attention,
normalization, and reductions may differ at the level of floating-point rounding.

### Shape strictness

The most visible behavioral difference is broadcasting:

```python
import numpy as np

prediction = [[1.0, 2.0]]

try:
    N.Loss.loss_mse(prediction, [0.0, 0.0])
except ValueError as error:
    print("native:", error)  # Elementwise operands must have the same shape

print("numpy:", N.Loss.loss_mse(np.array(prediction), [0.0, 0.0]))  # 2.5
```

The NumPy call returns a value by broadcasting the flat target against the $(1,2)$ prediction.
That value is usually not the reduction you intended. Always pass targets with the exact output
shape, for example `[[1.0, 2.0]]` instead of `[1.0, 2.0]`. Built-in reductions are the same in
both backends:

- MSE and BCE average over all output elements.
- CCE sums over classes and averages over rows.
- Gradients already include these factors.

See [losses and gradients](loss.md) for the definitions.

### Masking, loss, and gradients

`padding_mask`, causal masking, loss definitions, and custom gradient callbacks behave the same
on both backends. Gradient callbacks receive nested lists for the native backend and NumPy
arrays for the NumPy backend; return the same representation the callback received, or a value
that `N.Math.array(...)` can convert.

## Serialization and portability

JSON files are backend-independent. The file stores plain numbers and does not record which
backend wrote it:

```python
model.save("model.json")

# Load with either backend.
native = N.load("model.json", backend=N.Backends.PYTHON)
numpy_model = N.load("model.json", backend=N.Backends.NUMPY)
```

Both models start with the same architecture and parameter values. Their forward passes agree up
to floating-point tolerance. If `backend` is omitted, `N.load()` uses the default NumPy
backend, which requires NumPy.

To continue training after loading, attach an optimizer, run a fresh forward, and call
`backward()`. Optimizer state is never included in JSON. See
[serialization](serialization.md) for the complete list of omitted state.

## Testing and NumPy-free operation

The full test suite is normally run with NumPy installed:

```sh
python -m unittest discover -s tests -v
```

The suite includes a check that the native backend works when NumPy cannot be imported. The
native backend itself never calls NumPy, so a model constructed with
`N(backend=N.Backends.PYTHON)` remains usable in a standard-library-only environment.

Some examples and utilities use NumPy for convenience. For example, the Titanic notebook calls
NumPy methods on predictions. Those calls are application code, not requirements of the native
backend. See [examples and troubleshooting](examples-and-troubleshooting.md#titanic-classification).

## Choosing a backend

| Situation | Recommendation |
|---|---|
| NumPy is installed and performance matters | NumPy |
| NumPy is unavailable | Native Python |
| Learning or debugging internals | Native Python |
| Inspecting exact nested-list values | Native Python |
| Training a sequence model on a larger batch | NumPy |
| Reproducing a model across environments | Save JSON and load with the backend available there |
| Comparing implementations numerically | Build once, convert or reload, allow tolerance |

Neither backend provides GPU acceleration, automatic differentiation, or full vectorization of
every layer. The native backend prioritizes portability and readability; the NumPy backend
prioritizes matrix-operation speed.

## Backend-related API

| Call | Purpose |
|---|---|
| `N.Backends.PYTHON`, `N.Backends.NUMPY` | Backend enum values |
| `N.backends` | Alias of `N.Backends` |
| `N(backend=...)` | Select at construction; required before NumPy is imported |
| `model.backend(...)` | Convert the model, reset optimizer state, clear caches |
| `model._backend` | Active enum (private implementation detail) |
| `N.Math.numpy()` | Import and return NumPy, raising `ImportError` if unavailable |
| `N.Math.is_numpy(value)` | Whether a value is a NumPy object |
| `N.Math.array(value, backend)` | Copy a value into the selected representation |
| `N.load(path, backend=...)` | Load JSON with the selected backend |

For layer-level behavior, continue with the [layer guides](layers/README.md). For the execution
model shared by both backends, see [architecture](architecture.md).