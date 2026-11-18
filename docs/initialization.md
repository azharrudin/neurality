# Weight initialization

[Documentation index](README.md) · [Dense layer](layers/dense.md) · [Attention](layers/attention.md) ·
[Architecture](architecture.md)

Initialization determines the parameter values before the first forward pass. Neurality
provides three dense-layer strategies plus fixed per-layer defaults for sequence components.
Random initializers are normal distributions; the scale argument is a standard deviation.

## Dense layers

Pass an initializer to `dense()`:

```python
model.dense(64, N.Activations.relu, initialization=N.Initializations.kaiming)
model.dense(32, N.Activations.tanh, initialization=N.Initializations.glorot)
model.dense(10, N.Activations.softmax)  # default initialization
```

For a layer with fan-in $D$ (input features) and fan-out $C$ (neurons), each weight is drawn
independently:

| `initialization` | Distribution | Standard deviation |
|---|---|---|
| `None` | $\mathcal{N}(0,\,0.1^2)$ | $0.1$ |
| `N.Initializations.kaiming` | $\mathcal{N}\!\left(0,\,\frac{2}{D}\right)$ | $\sqrt{\frac{2}{D}}$ |
| `N.Initializations.glorot` | $\mathcal{N}\!\left(0,\,\frac{2}{D+C}\right)$ | $\sqrt{\frac{2}{D+C}}$ |
| custom callable | whatever it returns | validated per neuron |

These are **normal** variants, not the uniform variants sometimes associated with the same
names. For example, Kaiming initialization here draws
$w \sim \mathcal{N}(0, 2/D)$, producing variance $2/D$.

The bias of every neuron starts at the `bias` argument of `dense()`, which defaults to $0$ and
must be finite. Bias initialization has no separate strategy.

### Choosing a strategy

| Activation | Common pairing |
|---|---|
| ReLU, Leaky ReLU | Kaiming |
| Tanh, sigmoid, linear | Glorot or the default |
| Softmax output | Default or Glorot |

The library does not select an initializer based on the activation automatically. `None` always
means normal with standard deviation $0.1$.

## Custom initializers

A custom initializer is a callable with the signature

```python
def initializer(fan_in, fan_out):
    ...
    return weights
```

It must return a finite vector of length `fan_in` for one neuron. `dense()` calls it once **per
neuron**, so each call should produce a fresh vector:

```python
import random

def scaled_normal(fan_in, fan_out):
    return [random.gauss(0.0, 0.05) for _ in range(fan_in)]

model.dense(16, N.Activations.tanh, initialization=scaled_normal)
```

The return value is validated:

- its shape must be exactly `(fan_in,)`;
- every value must be finite.

If the initializer returns the same vector on every call, all neurons in the layer will start
identical and receive different gradients only because of their inputs. Depending on the task,
that symmetry may or may not break.

Custom initializers are not stored in JSON; only the resulting weights are saved.

## Sequence-layer defaults

Sequence layers do not accept an initialization argument. They use fixed distributions:

| Layer | Parameter | Shape | Distribution |
|---|---|---|---|
| Token embedding | $E$ | $(V,D)$ | $\mathcal{N}\!\left(0,\frac{1}{D}\right)$ |
| Positional embedding | $P$ | $(L_{\max},D)$ | $\mathcal{N}\!\left(0,\frac{1}{D}\right)$ |
| Attention | $W_Q, W_K$ | $(D,d_k)$ | $\mathcal{N}\!\left(0,\frac{2}{D+d_k}\right)$ |
| Attention | $W_V$ | $(D,D)$ | $\mathcal{N}\!\left(0,\frac{1}{D}\right)$ |
| Multi-head attention | $W_Q,W_K,W_V,W_O$ | $(D,D)$ | $\mathcal{N}\!\left(0,\frac{1}{D}\right)$ |
| Layer normalization | $\gamma$ | $(D,)$ | all ones |
| Layer normalization | $\beta$ | $(D,)$ | all zeros |

Attention projections have no bias parameters. Layer normalization starts as an identity
affine transform after normalization.

## Reproducibility

Random initialization uses the generator of the model's backend:

| Backend | Generator | Seed with |
|---|---|---|
| `PYTHON` | `random.gauss` | `random.seed(value)` |
| `NUMPY` | `numpy.random.randn` | `numpy.random.seed(value)` |

Seed the relevant generator before building layers:

```python
import random
import numpy as np

random.seed(42)
np.random.seed(42)

model = N(backend=N.Backends.NUMPY)
model.dense(8, N.Activations.tanh)
```

Matching integer seeds do **not** produce identical weights across the two generators. For a
numeric comparison across backends, build one model and reload or convert its parameters rather
than reinitializing it.

`N.Math.randn()` follows the global `Neurality._default_backend`, not the backend of a
particular model. Built-in initializers passed to `dense()` are special-cased to use the model's
backend. Custom initializers choose their own source of randomness. See
[backends](backends.md#randomness-and-initialization) for the full backend comparison.

## Related guides

- [Dense layer](layers/dense.md)
- [Attention](layers/attention.md)
- [Multi-head attention](layers/multihead-attention.md)
- [Training: reproducibility](training.md#reproducibility-and-backend-changes)
- [Serialization](serialization.md)