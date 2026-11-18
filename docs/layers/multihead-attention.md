# Multi-head self-attention

[Documentation index](../README.md) · [Layer overview](README.md) · [Attention](attention.md) ·
[Transformer guide](../transformer.md)

Multi-head attention runs several smaller attention operations in parallel on different
learned projections, concatenates their outputs, and mixes them with a learned output
projection. It is the attention layer used in the Transformer example.

```python
model.multihead_attention(num_heads, causal=False, embedding_size=None)
```

The method returns the model.

## Arguments

| Argument | Type | Default | Rules |
|---|---|---|---|
| `num_heads` | `int` | required | Positive and must divide the feature width $D$ |
| `causal` | `bool` | `False` | If true, position $i$ may only attend to $j \le i$ |
| `embedding_size` | `int` or `None` | Inferred | Feature width when this is the first layer |

`embedding_size` follows the same rules as in [single-head attention](attention.md#arguments):
it sets `INPUT_SIZE` for a first layer, and must match the preceding width otherwise.

Each head has width

$$
d_h = \frac{D}{H}.
$$

Because `num_heads` must divide $D$, all heads have equal width. An invalid head count raises
`ValueError`.

## Shapes

| Input shape | Interpretation | Output shape |
|---|---|---|
| $(T,D)$ | One sequence | $(T,D)$ |
| $(B,T,D)$ | Batch of independent sequences | $(B,T,D)$ |

The output width is $D$, matching the input. A 2D input is one sequence; rows within it attend
to each other.

## Forward pass

The layer learns four projection matrices, all of shape $(D,D)$:

$$
W_Q,\ W_K,\ W_V,\ W_O \in \mathbb{R}^{D \times D}.
$$

First it projects the input:

$$
Q = XW_Q,
\qquad
K = XW_K,
\qquad
V = XW_V,
\qquad
Q,K,V \in \mathbb{R}^{T \times D}.
$$

The projections are split into $H$ contiguous heads of width $d_h$. With 1-based block notation,

$$
Q_h = Q\left[:,\ (h-1)d_h + 1 : h d_h\right],
$$

and likewise $K_h$ and $V_h$. Each head performs scaled dot-product attention with its own
scale:

$$
S_h = \frac{Q_h K_h^\top}{\sqrt{d_h}} \in \mathbb{R}^{T \times T},
$$

$$
P_h = \operatorname{softmax}_{\mathcal{A}}(S_h),
\qquad
O_h = P_h V_h \in \mathbb{R}^{T \times d_h}.
$$

The allowed-key set $\mathcal{A}_i$ is identical to single-head attention: it combines the
padding mask and the causal condition $j \le i$. Masked entries receive probability zero; an
empty allowed set produces a zero row.

Head outputs are concatenated along the feature axis:

$$
C = \left[\,O_1 \;\middle|\; O_2 \;\middle|\; \cdots \;\middle|\; O_H\,\right]
\in \mathbb{R}^{T \times D}.
$$

Finally the output projection mixes the heads:

$$
Y = C W_O \in \mathbb{R}^{T \times D}.
$$

## Padding masks

The same `padding_mask` passed to `forward()` applies to every head of every multi-head layer.
It excludes padded positions as keys and values; it does not mask queries, embeddings, or
losses. The full mask contract is documented in
[single-head attention](attention.md#padding-masks) and
[Transformer mask semantics](../transformer.md#mask-semantics).

## Backward pass

Let $dY = \partial L / \partial Y$ for one sequence. The output projection receives

$$
dW_O = C^\top dY,
\qquad
dC = dY\,W_O^\top.
$$

Split $dC$ into head slices $dC_h$ using the same contiguous blocks as the forward pass. Each
head is differentiated exactly like a single-head attention layer of width $d_h$:

$$
dV_h = P_h^\top dC_h,
\qquad
dP_h = dC_h\,V_h^\top,
$$

$$
dS_h = \frac{1}{\sqrt{d_h}}\,P_h \odot
\left(dP_h - \langle P_h, dP_h\rangle\,\mathbf{1}^\top\right),
$$

where the inner product is taken row by row,

$$
\langle P_h, dP_h\rangle_i = \sum_{k=1}^{T} P_{h,ik}\, dP_{h,ik}.
$$

Then

$$
dQ_h = dS_h K_h,
\qquad
dK_h = dS_h^\top Q_h.
$$

Reassembling the head gradients along the feature axis gives $dQ$, $dK$, and $dV$, each of
shape $(T,D)$. Projection gradients are

$$
dW_Q = X^\top dQ,
\qquad
dW_K = X^\top dK,
\qquad
dW_V = X^\top dV.
$$

The input gradient sums the contribution of every head:

$$
dX = \sum_{h=1}^{H}
\left(
dQ_h\,W_{Q,h}^\top
+ dK_h\,W_{K,h}^\top
+ dV_h\,W_{V,h}^\top
\right),
$$

where $W_{Q,h}$ denotes the columns of $W_Q$ that produce head $h$, and similarly for keys and
values.

For a batch, the parameter gradients are accumulated across sequences before the optimizer
step; $dX$ is assembled per sequence.

## Initialization

All four projections are initialized independently from

$$
W \sim \mathcal{N}\!\left(0, \frac{1}{D}\right),
$$

with standard deviation $1/\sqrt{D}$. There are no attention biases. This differs from
single-head attention, where query/key projections use $\sqrt{2/(D+d_k)}$.

## Example

```python
import random
from lib.neurality import Neurality as N

random.seed(7)
model = N(backend=N.Backends.PYTHON)
model.embedding(vocab_size=12, embedding_size=8)
model.positional_embedding(max_length=16)
model.multihead_attention(num_heads=2, causal=True)

batch = [[1, 2, 3, 4], [4, 3, 2, 1]]
output = model.forward(batch)
print(N.Math.shape(output))  # (2, 4, 8)
```

With a padding mask:

```python
output = model.forward(
    [[1, 2, 3, 0], [4, 5, 0, 0]],
    padding_mask=[[1, 1, 1, 0], [1, 1, 0, 0]],
)
print(N.Math.shape(output))  # (2, 4, 8)
```

## Comparison with single-head attention

| Property | `attention()` | `multihead_attention()` |
|---|---|---|
| Projections | $W_Q, W_K, W_V$ | $W_Q, W_K, W_V, W_O$ |
| $W_Q, W_K$ shape | $(D,d_k)$ | $(D,D)$ |
| $W_V$ shape | $(D,D)$ | $(D,D)$ |
| Output projection | none | $W_O$ |
| Scale | $1/\sqrt{d_k}$ | $1/\sqrt{D/H}$ |
| Head count | 1 | $H$ |
| JSON `type` | `attention` | `multihead_attention` |

## Cost and constraints

- **Quadratic attention.** Scores and probabilities are $T \times T$ for each of $H$ heads:
  $O(BHT^2)$ values cached per batch. Time grows as $O(BHT^2 d_h)$.
- **No dropout or biases.** The layer implements projections, masking, softmax, concatenation,
  and output projection only.
- **No cross-attention or KV cache.** Queries, keys, and values share the same input, and every
  forward recomputes the full context.
- **Head count divides width.** A width that is not divisible by `num_heads` is rejected at
  construction.
- **Causal masks are per layer.** Set `causal=True` on each layer that must not see the future.

## Serialization

JSON stores `type`, `embedding_size`, `num_heads`, `causal`, and the four matrices `Wq`, `Wk`,
`Wv`, `Wo`. Loading validates that `num_heads` divides the width, that `causal` is a boolean,
that all matrices have shape $(D,D)$, and that all values are finite.

## Related guides

- [Single-head attention](attention.md)
- [Layer normalization](normalization.md)
- [Residual connections](residual.md)
- [Transformer guide](../transformer.md)
- [Serialization](../serialization.md)