# Token embedding layer

[Documentation index](../README.md) · [Layer overview](README.md) ·
[Positional embedding](positional-embedding.md) · [Serialization](../serialization.md)

A token embedding maps integer token IDs to trainable feature vectors. The layer owns a lookup
table with one row per vocabulary entry. It is normally the first layer of a token or language
model.

```python
model.embedding(vocab_size, embedding_size)
```

The method returns the model.

## Arguments

| Argument | Type | Rules |
|---|---|---|
| `vocab_size` | `int` | Positive; number of rows $V$ |
| `embedding_size` | `int` | Positive; feature width $D$ |

`embedding()` must be the first layer. Calling it after any other layer raises `ValueError`.
Both arguments must be positive integers; booleans are rejected because the check requires
`type(size) is int`.

## Shapes

| Input shape | Meaning | Output shape |
|---|---|---|
| $(T,)$ | One sequence of IDs | $(1,T,D)$ |
| $(B,T)$ | Batch of sequences | $(B,T,D)$ |

The table has shape

$$
E \in \mathbb{R}^{V \times D}.
$$

Input values must be integral and lie in $[0, V)$. They may be Python integers or floating-point
numbers with an exact integer value, because the forward pass checks `v == int(v)` and then
converts to `int`. A fractional value such as `1.5` is rejected, as is any ID outside the
vocabulary.

A 1D input is promoted to a batch of one sequence by `forward()`, so `[2, 7, 3]` produces shape
$(1,3,D)$.

## Forward pass

For a batch of token sequences $x \in \mathbb{Z}^{B \times T}$:

$$
H_{b,t,:} = E_{x_{b,t},:}.
$$

That is, each token ID selects one row of the table. There is no bias, activation, scaling, or
projection in this layer. The output already has feature width $D$, so a following
`positional_embedding`, `norm`, `attention`, or dense layer uses $D$ as its input width.

The forward pass caches the validated integer IDs for backward. It does not cache a copy of the
table rows themselves.

## Backward pass

There is no gradient with respect to a discrete token ID. The incoming gradient
$dH \in \mathbb{R}^{B \times T \times D}$ is routed to the table rows that were used:

$$
\frac{\partial L}{\partial E_v}
= \sum_{(b,t)\,:\,x_{b,t}=v} dH_{b,t,:}.
$$

Every occurrence of token $v$ contributes to the same row, so rows used more often receive
larger accumulated gradients. Rows that never appear in the batch receive a zero gradient and
do not change that step.

The layer appends one gradient matrix of shape $(V,D)$ to the optimizer's parameter list.
No gradient is passed to a preceding layer, because this layer must be first.

## Initialization

Table entries are drawn independently from

$$
E_{v,j} \sim \mathcal{N}\!\left(0, \frac{1}{D}\right).
$$

The standard deviation is $1/\sqrt{D}$. Initialization uses the model's backend generator:
Python's `random.gauss` for the pure-Python backend and `numpy.random.randn` for NumPy.
Seeding before construction makes initialization reproducible on that backend.

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

A complete token model adds attention, normalization, residuals, and an output projection. See
[Attention and Transformers](../transformer.md).

## Padding, tokenizers, and freezing

- **No reserved padding ID.** The layer does not treat any ID specially. To pad sequences, choose
  an ID in $[0,V)$ and mark padding positions with `padding_mask` when calling `forward()`. The
  mask affects attention keys and values, not the embedding lookup.
- **No tokenizer.** The library accepts IDs. Mapping text to IDs belongs to application code.
  The BPE tokenizer in the Transformer notebook is example code. Its vocabulary and merge order
  must be persisted separately from the model.
- **No frozen embeddings.** The table is always trainable during `backward()`.
- **No weight tying.** The embedding table is independent of any output projection, even if both
  have shape $(V,D)$.

Changing the tokenizer after training changes the meaning of each row. A saved model therefore
means nothing without the exact token-to-ID mapping used during training.

## Serialization

The JSON layer stores `type`, `vocab_size`, `embedding_size`, and `weights` (a nested array of
shape $(V,D)$). Loading validates that the table has exactly that shape, that values are finite,
and that the embedding is the first layer.

## Related guides

- [Positional embedding](positional-embedding.md)
- [Attention](attention.md)
- [Mask semantics](../transformer.md#mask-semantics)
- [Serialization](../serialization.md)