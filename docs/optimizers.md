# Optimizers

[Documentation index](README.md) · [Architecture](architecture.md) · [Losses](loss.md) ·
[Training](training.md)

An optimizer turns parameter gradients into parameter updates. Neurality implements four
first-order optimizers. Every one applies

$$
\theta \leftarrow \theta - \eta\,u(g),
$$

where $\theta$ is a parameter, $g = \partial L/\partial\theta$ is its gradient, $\eta$ is the
learning rate, and $u(g)$ is the optimizer-specific update direction.

## Available optimizers

| Class | Update direction $u$ |
|---|---|
| `N.Optimizers.SGD` | $g$ |
| `N.Optimizers.Momentum` | Smoothed velocity $v$ |
| `N.Optimizers.RMSprop` | $g/(\sqrt{s}+\epsilon)$ |
| `N.Optimizers.Adam` | Bias-corrected first moment divided by corrected second-moment root |

### SGD

$$
u = g.
$$

Plain stochastic gradient descent. There is no weight decay, Nesterov term, or learning-rate
schedule.

### Momentum

$$
v \leftarrow \mu v + g,
\qquad
u = v.
$$

$v$ starts at zero and is stored per parameter. This is the classical momentum form without a
$(1-\mu)$ factor on the gradient. Larger $\mu$ carries more history.

### RMSprop

$$
s \leftarrow \rho s + (1-\rho)g^2,
\qquad
u = \frac{g}{\sqrt{s}+\epsilon}.
$$

$s$ is an exponentially decaying average of squared gradients. It starts at zero. The
per-parameter scaling adapts the step size to gradient magnitude.

### Adam

$$
m \leftarrow \beta_1 m + (1-\beta_1)g,
$$

$$
v \leftarrow \beta_2 v + (1-\beta_2)g^2,
$$

$$
\hat{m} = \frac{m}{1-\beta_1^t},
\qquad
\hat{v} = \frac{v}{1-\beta_2^t},
$$

$$
u = \frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}.
$$

$m$ and $v$ start at zero. The bias corrections use the optimizer's step counter $t$, which
increments once per `backward()` call (one parameter update), not once per parameter. All
parameters in that update use the same $t$.

## Constructors and defaults

| Constructor | Defaults | Validation |
|---|---|---|
| `SGD(learning_rate=0.01)` | $\eta = 0.01$ | $\eta > 0$, finite |
| `Momentum(learning_rate=0.01, momentum=0.9)` | $\eta = 0.01$, $\mu = 0.9$ | $\eta > 0$ finite; $0 \le \mu < 1$ |
| `RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8)` | $\eta = 0.001$, $\rho = 0.9$, $\epsilon = 10^{-8}$ | $\eta > 0$ finite; $0 \le \rho < 1$; $\epsilon > 0$ finite |
| `Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)` | $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ | $\eta > 0$ finite; $0 \le \beta_1,\beta_2 < 1$; $\epsilon > 0$ finite |

Invalid values raise `ValueError` at construction time.

### The default optimizer is not `SGD()`

A new model starts with

```python
N.Optimizers.SGD(learning_rate=0.1)
```

whereas explicitly calling `N.Optimizers.SGD()` constructs a rate of `0.01`. This difference is
easy to miss. Set the optimizer explicitly when the rate matters.

## Attaching an optimizer

```python
model.optimizer(N.Optimizers.Adam(learning_rate=0.03))
```

`model.optimizer(...)` accepts an instance of `SGD` or any subclass. It **copies** the
optimizer's configuration into the model and resets its state. Accumulated moments from the
original instance are not carried over. The source instance and the model's copy start with
empty state; avoid continuing to use the source instance as a separate optimizer afterward.

Call it once before training. Repeated calls discard all moments and step counts:

```python
model.optimizer(N.Optimizers.Adam(learning_rate=0.03))
# ... train ...
model.optimizer(N.Optimizers.Adam(learning_rate=0.03))  # starts from zero state
```

## Per-update learning-rate override

`backward()` can override the learning rate for one update:

```python
model.backward(target, N.Gradients.gradient_bce, learning_rate=0.005)
```

The override must be positive and finite. It does not change the optimizer's stored
`learning_rate`, so the next `backward()` uses the configured value again. This is useful for
one-off probes, but there is no built-in schedule.

## Update mechanics

`backward()` does not call the optimizer until it has traversed every layer and collected all
parameter gradients. Only then does it invoke one optimizer step with the full list. Therefore:

- All gradients are computed at the same parameter state.
- One `backward()` produces exactly one update for every trainable parameter.
- The optimizer's step counter advances once per `backward()`.
- There is no gradient accumulation, mini-batch accumulation, or gradient clipping.

Parameters updated include:

| Layer | Updated values |
|---|---|
| Dense | Each neuron's `w` and `b` |
| Embedding | The full table |
| Positional embedding | The full table |
| Norm | `gamma`, `beta` |
| Attention | `Wq`, `Wk`, `Wv` |
| Multi-head attention | `Wq`, `Wk`, `Wv`, `Wo` |
| Residual markers | none |

## State lifecycle

Optimizer state is kept in the optimizer object, keyed by the layer and parameter identity:

- SGD has no per-parameter state, only a step counter.
- Momentum stores one velocity per parameter.
- RMSprop stores one squared-gradient average per parameter.
- Adam stores two moments per parameter.

State is cleared when:

- `model.optimizer(...)` attaches a new optimizer;
- the same optimizer class is attached again;
- `model.backend(...)` converts the model;
- a model is loaded from JSON (the new model gets a fresh default optimizer).

State is **not** saved by `model.save()`. A resumed run therefore restarts momentum, RMSprop
averages, and Adam moments from zero. Rearranging the layer list or loading a different
architecture also invalidates the meaning of stored state, which is why the library resets it
rather than trying to migrate it.

## Choosing an optimizer

| Situation | Reasonable starting point |
|---|---|
| Small educational example | Adam with $\eta = 0.01$–$0.03$ |
| Simple convex-like objective | SGD, possibly Momentum |
| Uneven gradient scales across parameters | RMSprop or Adam |
| Reproducing a plain gradient-descent demonstration | SGD |

These are starting points, not guaranteed settings. There is no automatic learning-rate finder
or scheduler. If loss diverges, check target shapes, activation/loss pairing, and input scaling
before changing the optimizer.

## Example

```python
import random
from lib.neurality import Neurality as N

random.seed(42)
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [1.0]]

model = N(backend=N.Backends.PYTHON).fit(X, y)
model.dense(4, N.Activations.tanh, initialization=N.Initializations.glorot)
model.dense(1, N.Activations.sigmoid, initialization=N.Initializations.glorot)
model.optimizer(N.Optimizers.Adam(learning_rate=0.03))

for epoch in range(500):
    prediction = model.forward(X)
    loss = N.Loss.loss_bce(prediction, y)
    model.backward(y, N.Gradients.gradient_bce)
```

## Related guides

- [Losses and gradients](loss.md)
- [Training and evaluation](training.md)
- [Architecture: optimizer step](architecture.md#optimizer-step)
- [Serialization: omitted state](serialization.md#what-is-omitted)
- [API reference](api.md#optimizers)