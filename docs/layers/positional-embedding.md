# Positional embedding layer

[Documentation index](../README.md) · [Layer overview](README.md) · [Embedding](embedding.md) ·
[Attention](attention.md)

Self-attention has no inherent notion of order. A learned positional embedding adds a trainable
vector for each sequence position so that the model can distinguish position $t=0$ from $t=1$.

```python
model.positional_embedding(max_length)
```

The method returns the model. It takes no feature-width argument: the width follows the
preceding layer, or `INPUT_SIZE` when it is the first layer.

## Arguments

| Argument | Type | Rules |
|---|---|---|
| `max_length` | `int` | Positive; maximum number of positions $L_{\max}$ |

The resulting table has shape

$$
P \in \mathbb{R}^{L_{\max} \times D},
$$

where $D$ is `model._feature_size()` at construction time. If the layer is first, that value is
`model.INPUT_SIZE`; set it with `fit()` or an `embedding_size` argument on a preceding attention
layer.

## Shapes

The layer preserves the input shape:

| Input shape | Output shape |
|---|---|
| $(T,D)$ | $(T,D)$ |
| $(B,T,D)$ | $(B,T,D)$ |

It requires $T \le L_{\max}$. A longer sequence raises
`ValueError("sequence exceeds positional embedding max_length")`.

## Forward pass

For each sequence position $t$:

$$
H_{b,t,:} \leftarrow H_{b,t,:} + P_{t,:},
\qquad t = 0, 1, \dots, T-1.
$$

Positions are numbered from zero on every forward call. There is no position-offset argument, so
a windowed context that starts at a later token still receives position vectors starting at
$P_0$.

The addition occurs after any preceding embedding layer, so a typical prefix is

```python
model.embedding(vocab_size, embedding_size).positional_embedding(max_length)
```

## Backward pass

The layer has no nonlinearity and no per-position mixing. The input gradient is therefore
unchanged:

$$
dH_{b,t,:} = \frac{\partial L}{\partial H_{b,t,:}}.
$$

The table gradient sums contributions from every sequence and every occurrence of a position:

$$
\frac{\partial L}{\partial P_t}
= \sum_{b=1}^{B} dH_{b,t,:}.
$$

The optimizer receives one gradient matrix of shape $(L_{\max}, D)$. Rows beyond the longest
sequence in the batch get zero gradient for that step.

## Initialization

Each entry is drawn independently from

$$
P_{t,j} \sim \mathcal{N}\!\left(0, \frac{1}{D}\right),
$$

with standard deviation $1/\sqrt{D}$, using the model's backend generator. Pos vectors are
learned; there is no sinusoidal or fixed encoding option.

## Example

```python
import random
from lib.neurality import Neurality as N

random.seed(7)
model = N(backend=N.Backends.PYTHON)
model.embedding(vocab_size=10, embedding_size=8)
model.positional_embedding(max_length=4)

output = model.forward([[2, 3, 0], [4, 0, 0]])
print(N.Math.shape(output))  # (2, 3, 8)
```

## Behavior and constraints

- **Capacity is fixed at construction.** To support longer sequences, build and train a model
  with a larger `max_length`.
- **Positions restart at zero.** During autoregressive generation with a truncated context, the
  retained tokens are re-numbered from zero; they do not keep their original absolute positions.
- **No masking interaction.** Padded positions receive a position vector like any other
  position. The padding mask only affects attention keys and values.
- **First-layer width.** When used first, the width comes from `INPUT_SIZE` (default 2) unless
  changed. For token models, place it after `embedding()` so the width is unambiguous.
- **Serialization.** JSON stores `type`, `embedding_size`, `max_length`, and `weights` of shape
  $(L_{\max},D)$. Loading checks that the width matches the preceding layer and that
  `max_length` is a positive integer.

## Related guides

- [Token embedding](embedding.md)
- [Attention](attention.md)
- [Transformer guide](../transformer.md)
- [Serialization](../serialization.md)