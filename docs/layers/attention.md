# Single-head self-attention

[Documentation index](../README.md) · [Layer overview](README.md) ·
[Multi-head attention](multihead-attention.md) · [Transformer guide](../transformer.md)

Single-head self-attention lets every position in a sequence build its output from a
weighted combination of all allowed positions. The layer learns three projections — query,
key, and value — and has no separate output projection.

```python
model.attention(key_size=None, causal=False, embedding_size=None)
```

The method returns the model.

## Arguments

| Argument | Type | Default | Rules |
|---|---|---|---|
| `key_size` | `int` or `None` | Feature width | Positive; width $d_k$ of queries and keys |
| `causal` | `bool` | `False` | If true, position $i$ may only attend to $j \le i$ |
| `embedding_size` | `int` or `None` | Inferred | Feature width when this is the first layer |

`embedding_size` handling:

- When supplied and the layer is first, it sets `model.INPUT_SIZE = embedding_size`.
- After another layer, it must equal the preceding layer's output width, otherwise a
  `ValueError` is raised.
- When omitted, the width is the preceding output width, or `INPUT_SIZE` for a first layer.

`causal` must be a real `bool`; `0`, `1`, and truthy objects are rejected.

## Shapes

| Input shape | Interpretation | Output shape |
|---|---|---|
| $(T,D)$ | One sequence | $(T,D)$ |
| $(B,T,D)$ | Batch of independent sequences | $(B,T,D)$ |

Attention preserves the feature width $D$. It does not change batch size or sequence length.

A 2D feature matrix is always treated as **one sequence**, even if its rows look like
independent samples. Passing a table of unrelated tabular rows to attention makes each row
attend to the others. Use 3D input to keep sequences independent.

## Forward pass

For one sequence $X \in \mathbb{R}^{T \times D}$, the layer learns

$$
W_Q \in \mathbb{R}^{D \times d_k},
\qquad
W_K \in \mathbb{R}^{D \times d_k},
\qquad
W_V \in \mathbb{R}^{D \times D}.
$$

It computes queries, keys, and values:

$$
Q = XW_Q,
\qquad
K = XW_K,
\qquad
V = XW_V.
$$

The scaled dot-product scores are

$$
S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}.
$$

Scores are converted to attention probabilities with a masked softmax over allowed keys. For
query position $i$, define the allowed key set

$$
\mathcal{A}_i = \{\, j : \text{mask}_j = 1 \ \text{and}\ (\text{not causal} \ \text{or}\ j \le i) \,\}.
$$

Then

$$
P_{ij} =
\begin{cases}
\dfrac{\exp(S_{ij})}{\sum_{k \in \mathcal{A}_i} \exp(S_{ik})}, & j \in \mathcal{A}_i, \\[6pt]
0, & j \notin \mathcal{A}_i.
\end{cases}
$$

For numerical stability, the implementation subtracts $\max_{k \in \mathcal{A}_i} S_{ik}$ before
exponentiation. If $\mathcal{A}_i$ is empty — possible when every position is padding, or when
causal masking starts beyond all valid keys — the entire probability row is zero.

The output is

$$
O = PV \in \mathbb{R}^{T \times D}.
$$

There is no output projection in this layer. If you need one, use
[multi-head attention](multihead-attention.md) or add a dense layer.

## Padding masks

`forward(data, padding_mask=...)` supplies the mask to every attention layer in the model.

| Property | Behavior |
|---|---|
| Accepted values | `1`/`True` for valid, `0`/`False` for padding |
| Shape | $(B,T)$, or $(T,)$ when the model processes one sequence |
| Scope | Applied to every attention layer in the forward pass |
| Effect | Excludes padded positions as **keys and values** |
| Not affected | Padded **queries**, embedding lookups, losses, targets |

A padded query can still attend to valid earlier keys and produce a nonzero output. Conversely,
a valid query cannot attend to a padded key. The mask is not an arbitrary pairwise attention
matrix; there is no general `(T,T)` mask parameter.

Embedding rows are not frozen or zeroed by selecting a padding ID. The mask decides validity.

The built-in losses do not consult the mask. Padded positions still contribute to loss values
unless application code supplies a masked loss and gradient. See
[loss masking](../transformer.md#loss-masking).

## Backward pass

Let $dO = \partial L / \partial O$ for one sequence. Values and probabilities receive

$$
dV = P^\top dO,
\qquad
dP = dO\,V^\top.
$$

The masked softmax is differentiated with a row-wise Jacobian–vector product. For each row
$i$,

$$
dS_{ij} = \frac{1}{\sqrt{d_k}} P_{ij}\left(dP_{ij} - \sum_{k=1}^{T} P_{ik} dP_{ik}\right).
$$

Because $P_{ij}=0$ for masked entries and for an empty allowed set, their score gradients are
zero as well.

Query and key gradients follow from the score definition:

$$
dQ = dS\,K,
\qquad
dK = dS^\top Q.
$$

The projection gradients for one sequence are

$$
dW_Q = X^\top dQ,
\qquad
dW_K = X^\top dK,
\qquad
dW_V = X^\top dV.
$$

For a batch, these parameter gradients are summed over sequences before the optimizer step. The
input gradient for each sequence is

$$
dX = dQ\,W_Q^\top + dK\,W_K^\top + dV\,W_V^\top,
$$

and the per-sequence results are returned as a batch.

## Initialization

| Parameter | Shape | Distribution |
|---|---|---|
| $W_Q$ | $(D,d_k)$ | $\mathcal{N}\!\left(0,\ \frac{2}{D+d_k}\right)$ |
| $W_K$ | $(D,d_k)$ | $\mathcal{N}\!\left(0,\ \frac{2}{D+d_k}\right)$ |
| $W_V$ | $(D,D)$ | $\mathcal{N}\!\left(0,\ \frac{1}{D}\right)$ |

Standard deviations are $\sqrt{2/(D+d_k)}$ for queries and keys and $1/\sqrt{D}$ for values.
There are no biases in attention projections.

## Example

```python
from lib.neurality import Neurality as N

model = N(backend=N.Backends.PYTHON)
model.attention(embedding_size=3, key_size=2)

sequence = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
output = model.forward(sequence)
print(N.Math.shape(output))  # (2, 3): one sequence, two positions

batched = model.forward([sequence, sequence])
print(N.Math.shape(batched))  # (2, 2, 3)
```

With a padding mask:

```python
model = N(backend=N.Backends.PYTHON)
model.embedding(10, 8).positional_embedding(4)
model.multihead_attention(num_heads=2, causal=True)

output = model.forward(
    [[2, 3, 0], [4, 0, 0]],
    padding_mask=[[1, 1, 0], [1, 0, 0]],
)
print(N.Math.shape(output))  # (2, 3, 8)
```

## Cost and constraints

- **Quadratic attention.** Scores and probabilities are $T \times T$ per sequence. Time and
  memory grow as $O(T^2 D)$ and $O(T^2)$ per sequence.
- **No dropout, bias, or output projection.** The layer is the minimal scaled dot-product
  attention.
- **No cross-attention.** Queries, keys, and values all come from the same input.
- **No KV cache.** Every forward recomputes the full context.
- **Causal masks are per layer.** Set `causal=True` on each attention layer that needs it.
- **Mask shape validation.** The mask must contain only `0`/`1` and match the batch and sequence
  dimensions inferred from the input; anything else raises `ValueError`.

## Serialization

JSON stores `type`, `embedding_size`, `key_size`, `causal`, and the three matrices `Wq`, `Wk`,
`Wv`. Loading checks that `embedding_size` matches the preceding layer, that `key_size` is
positive, that `causal` is a boolean, that the shapes are $(D,d_k)$, $(D,d_k)$, and $(D,D)$, and
that all values are finite.

## Related guides

- [Multi-head attention](multihead-attention.md)
- [Token embedding](embedding.md)
- [Mask semantics](../transformer.md#mask-semantics)
- [Serialization](../serialization.md)