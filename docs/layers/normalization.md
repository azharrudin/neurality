# Layer normalization

[Documentation index](../README.md) · [Layer overview](README.md) · [Residual connections](residual.md) ·
[Transformer guide](../transformer.md)

Layer normalization rescales each sample or token independently using statistics computed along
the final feature dimension. It has two trainable vectors, a scale $\gamma$ and a shift
$\beta$, and no dependence between rows.

```python
model.norm(epsilon=1e-5)
```

The method returns the model.

## Arguments

| Argument | Type | Default | Rules |
|---|---|---|---|
| `epsilon` | number | `1e-5` | Positive, finite, and not a boolean |

The feature width $D$ is inferred from the preceding layer, or from `INPUT_SIZE` when the layer
is first.

## Shapes

| Input shape | Output shape | Normalization axis |
|---|---|---|
| $(D,)$ | $(1,D)$ | the $D$ features |
| $(B,D)$ | $(B,D)$ | the $D$ features of each row |
| $(B,T,D)$ | $(B,T,D)$ | the $D$ features of each token |

Internally the layer flattens leading dimensions into rows, normalizes each row, and restores
the original shape.

Trainable parameters:

$$
\gamma \in \mathbb{R}^{D},
\qquad
\beta \in \mathbb{R}^{D}.
$$

$\gamma$ is initialized to all ones and $\beta$ to all zeros, so a freshly constructed layer
initially performs pure normalization.

## Forward pass

For one row $x \in \mathbb{R}^{D}$:

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i,
$$

$$
\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i - \mu)^2,
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}},
$$

$$
y_i = \gamma_i\,\hat{x}_i + \beta_i.
$$

Notes on the definition:

- The variance uses divisor $D$, not the sample correction $D-1$.
- $\epsilon$ is added inside the square root, before taking the reciprocal.
- Each row is normalized independently. There is no mixing across batch items or sequence
  positions.
- There are no running averages and no separate inference mode, so the layer behaves the same
  during training and evaluation.

The forward cache stores $\hat{x}$ and the inverse standard deviation
$1/\sqrt{\sigma^2+\epsilon}$ for every row. These are used in backward.

## Backward pass

Let $dy$ be the incoming gradient for one row. The affine parameters receive gradients summed
over all rows in the flattened batch:

$$
d\gamma = \sum_{r=1}^{R} dy_r \odot \hat{x}_r,
\qquad
d\beta = \sum_{r=1}^{R} dy_r.
$$

For the input, first backpropagate through the scale:

$$
d\hat{x} = dy \odot \gamma.
$$

Then, for each row independently,

$$
dx = \frac{1}{\sqrt{\sigma^2+\epsilon}}
\left[
d\hat{x}
- \overline{d\hat{x}}\,\mathbf{1}
- \hat{x} \odot \overline{\left(d\hat{x}\odot\hat{x}\right)}\,\mathbf{1}
\right],
$$

where the overbar denotes the mean over the $D$ features:

$$
\overline{u} = \frac{1}{D}\sum_{i=1}^{D} u_i.
$$

The two mean terms correct for the fact that $\mu$ and $\sigma^2$ both depend on $x$. The
implementation computes exactly this expression using the cached $\hat{x}$ and inverse standard
deviation.

## Example

```python
import random
from lib.neurality import Neurality as N

random.seed(3)
X = [[0.1, -0.2, 0.4, 0.7, 0.0, 0.3, -0.5, 0.2]]

model = N(backend=N.Backends.PYTHON).fit(X)
model.dense(8, N.Activations.linear)
model.norm(epsilon=1e-5)

output = model.forward(X)
print(N.Math.shape(output))  # (1, 8)
```

A pre-normalization block combines normalization with a residual connection:

```python
model.residual_start().norm()
model.multihead_attention(num_heads=2, causal=True)
model.residual_end()
```

## Behavior and constraints

- **Feature-axis only.** There is no option to normalize across batch or sequence axes.
- **No running statistics.** The layer computes fresh statistics on every forward call.
- **No dropout.** Nothing in this layer changes between training and inference.
- **Affine parameters are always trainable.** There is no `elementwise_affine=False` option.
- **Width must match.** The layer's width comes from its preceding layer. Inserting it before a
  layer with a different width requires a width-changing dense layer in between.
- **Serialization.** JSON stores `type`, `embedding_size`, `epsilon`, `gamma`, and `beta`.
  Loading validates the width, positive finite `epsilon`, and both parameter shapes.

## Related guides

- [Residual connections](residual.md)
- [Dense layer](dense.md)
- [Transformer guide](../transformer.md)
- [Initialization](../initialization.md)
- [Serialization](../serialization.md)