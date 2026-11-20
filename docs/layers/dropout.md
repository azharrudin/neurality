# Dropout

[Documentation index](../README.md) · [Layer overview](README.md) · [Training](../training.md) ·
[Transformer guide](../transformer.md)

Dropout randomly zeroes elements of its input during training and leaves the input unchanged
during evaluation. It is a regularization layer for reducing co-adaptation between features.
Neurality uses **inverted dropout**: kept values are divided by the keep probability during
training, so inference needs no rescaling.

```python
model.dropout(rate=0.1)
```

The method returns the model. Dropout has no trainable parameters.

## Arguments

| Argument | Type | Default | Rules |
|---|---|---|---|
| `rate` | number | `0.1` | Finite and in $[0, 1)$; booleans are rejected |

A rate of `0` is a no-op. The keep probability is

$$
q = 1 - \text{rate}.
$$

The feature width $D$ is inferred from the preceding layer, or from `INPUT_SIZE` when the layer
is first.

## Shapes

Dropout preserves shape:

| Input shape | Output shape |
|---|---|
| $(D,)$ | $(D,)$ or $(1,D)$ after model input promotion |
| $(B,D)$ | $(B,D)$ |
| $(B,T,D)$ | $(B,T,D)$ |

The mask has the same shape as the input.

## Forward pass

During training, each element $x_i$ is retained with probability $q$ and zeroed otherwise:

$$
m_i =
\begin{cases}
\dfrac{1}{q}, & \text{with probability } q,\\[4pt]
0, & \text{with probability } 1-q,
\end{cases}
\qquad
y_i = x_i\,m_i.
$$

The scaled mask $m$ is cached for backward. During evaluation, or when `rate == 0`, the layer is
the identity and no mask is drawn.

Dropout is active only when the model is in training mode. Toggle the mode with:

```python
model.train()   # dropout active
model.eval()    # dropout disabled
```

`forward(data, training=False)` also disables dropout for one call without changing the model
mode:

```python
prediction = model.forward(X, training=False)
```

## Backward pass

The dropout mask is linear and has no parameters. For incoming gradient $dy$:

$$
dx_i = dy_i\,m_i.
$$

The cached mask from the corresponding forward pass is reused, so the same elements that were
dropped receive zero gradient.

## Example

```python
import random
from lib.neurality import Neurality as N

random.seed(0)
X = [[1.0, 2.0], [3.0, 4.0]]

model = N(backend=N.Backends.PYTHON)
model.dropout(rate=0.5)

model.eval()
print(N.Math.tolist(model.forward(X)))  # identity

model.train()
print(N.Math.tolist(model.forward(X)))  # elements are 0 or 2*x
```

A pre-normalization Transformer block with dropout on the attention and feed-forward branches:

```python
model.dropout(0.1)                 # embedding dropout

model.residual_start().norm()
model.multihead_attention(num_heads=2, causal=True)
model.dropout(0.1)                 # attention output dropout
model.residual_end()

model.residual_start().norm()
model.dense(16, N.Activations.gelu)
model.dense(8, N.Activations.linear)
model.dropout(0.1)                 # feed-forward output dropout
model.residual_end()
```

## Behavior and constraints

- **Training-only.** Evaluation is always the identity; there are no running statistics.
- **Inverted scaling.** No rescaling is performed at inference.
- **Mask is cached.** `backward()` must follow the matching training `forward()`; do not run a
  validation forward between a training forward and its backward update.
- **No parameters.** JSON stores `type`, `embedding_size`, and `rate`.
- **Mode is runtime state.** The train/eval mode is not serialized; loaded models start in
  training mode, so call `model.eval()` before inference.
- **Serialization.** Loading validates the width and that `rate` is finite and in $[0, 1)$.
- **Padding.** Dropout applies to every element, including padded positions. Combine it with a
  masked loss if padded tokens must not influence training.

## Related guides

- [Training and evaluation](../training.md)
- [Normalization](normalization.md)
- [Transformer guide](../transformer.md)
- [Serialization](../serialization.md)