# Attention and Transformers

[Documentation index](README.md) · [Attention](layers/attention.md) ·
[Multi-head attention](layers/multihead-attention.md) · [Normalization](layers/normalization.md) ·
[Residuals](layers/residual.md)

Neurality provides the constituent layers of a Transformer, not a packaged Transformer block.
This guide shows how to assemble them, how masks behave, and how to train and sample from a
small language model.

## Sequence representation

Token IDs have shape $(B,T)$. `embedding(V,D)` maps them to $(B,T,D)$. Position vectors,
attention, normalization, and residual layers preserve this shape; a final dense softmax
produces $(B,T,V)$.

A 1D list of token IDs becomes one sequence. Without token embedding, attention accepts a
feature matrix $(T,D)$ as one sequence, or $(B,T,D)$ as independent sequences. A matrix of
independent tabular samples should not be passed to attention unless interaction between its
rows is intended.

```python
from lib.neurality import Neurality as N

model = N(backend=N.Backends.PYTHON)
model.attention(embedding_size=3, key_size=2)

output = model.forward([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
print(N.Math.shape(output))  # (2, 3): one sequence, two positions
```

## A complete small language model

This synthetic task teaches a repeating token cycle. It demonstrates the mechanics of a
pre-normalization Transformer-style block, rather than natural-language understanding.

```python
import random
from lib.neurality import Neurality as N

random.seed(7)
vocab_size = 4
max_length = 8

inputs = [[0, 1, 2, 3], [1, 2, 3, 0]]
next_tokens = [[1, 2, 3, 0], [2, 3, 0, 1]]
targets = [
    [[float(candidate == token) for candidate in range(vocab_size)]
     for token in sequence]
    for sequence in next_tokens
]

model = N(backend=N.Backends.PYTHON)
model.embedding(vocab_size, 8).positional_embedding(max_length)

# Attention block: y1 = x + MHA(LN(x))
model.residual_start().norm()
model.multihead_attention(num_heads=2, causal=True)
model.residual_end()

# Feed-forward block: y2 = y1 + FFN(LN(y1))
model.residual_start().norm()
model.dense(16, N.Activations.gelu, initialization=N.Initializations.glorot)
model.dense(8, N.Activations.linear, initialization=N.Initializations.glorot)
model.residual_end()

# Output head
model.norm().dense(vocab_size, N.Activations.softmax)
model.optimizer(N.Optimizers.Adam(learning_rate=0.01))

for epoch in range(100):
    prediction = model.forward(inputs)
    loss = N.Loss.loss_cce(prediction, targets)
    model.backward(targets, N.Gradients.gradient_cce)

print(f"Final pre-update loss: {loss:.6f}")

# Greedy autoregressive generation, continuing from the trained model.
generated = [0, 1]
for _ in range(6):
    context = generated[-max_length:]
    probabilities = model.forward([context])[0][-1]
    next_id = max(range(vocab_size), key=lambda i: probabilities[i])
    generated.append(next_id)

print(generated)
```

Input and target are shifted by one position. All positions contribute to CCE, which averages
over $B \cdot T$ rows. The feed-forward branch expands from 8 to 16 features and contracts to 8
before residual addition. The head count divides the model width. A final normalization precedes
vocabulary projection.

### The block in equations

Let $x \in \mathbb{R}^{T \times D}$ be the embedded sequence after positional addition, and let
$\mathrm{LN}$ denote layer normalization over the feature axis. The pre-normalization
attention sublayer computes

$$
y_1 = x + \mathrm{MHA}\!\left(\mathrm{LN}(x)\right),
$$

where

$$
\mathrm{MHA}(\cdot) = \left[\,
O_1 \;|\; \cdots \;|\; O_H
\,\right] W_O,
\qquad
O_h = \mathrm{softmax}\!\left(
\frac{Q_h K_h^\top}{\sqrt{d_h}}
\right)V_h,
$$

and $Q_h, K_h, V_h$ are the head slices of $XW_Q$, $XW_K$, $XW_V$. The feed-forward sublayer then
computes

$$
y_2 = y_1 + W_2\,\mathrm{GELU}\!\left(W_1\mathrm{LN}(y_1) + b_1\right) + b_2,
$$

with $W_1$ expanding $D \to D_{\text{ff}}$ (here $8 \to 16$) and $W_2$ projecting
$D_{\text{ff}} \to D$. Finally, the output head is

$$
p = \mathrm{softmax}\!\left(\mathrm{LN}(y_2)\,W_{\text{vocab}} + b_{\text{vocab}}\right),
$$

where $W_{\text{vocab}}$ is the dense vocabulary projection and $p \in \mathbb{R}^{T \times V}$.
Each row is a distribution over the next token.

Neurality has no dedicated `transformer_block()` constructor. The example composes the layers
directly. It has no dropout, weight tying, KV cache, or built-in sampling API.

## Mask semantics

Causal masking allows position $i$ to attend only to keys at positions $j \le i$. This prevents
the next-token prediction task from directly accessing future inputs in the same sequence.
`causal` is configured on each attention layer.

Padding masks are supplied to `forward()` and applied to every attention layer:

```python
from lib.neurality import Neurality as N

model = N(backend=N.Backends.PYTHON)
model.embedding(10, 8).positional_embedding(4)
model.multihead_attention(num_heads=2, causal=True)

output = model.forward(
    [[2, 3, 0], [4, 0, 0]],
    padding_mask=[[1, 1, 0], [1, 0, 0]],
)
print(N.Math.shape(output))  # (2, 3, 8)
```

Here `0` is an application-selected padding ID. Neurality does not reserve it; the mask
determines validity. Masks use $(B,T)$ with `1`/`True` for valid positions and `0`/`False` for
padding. One sequence also accepts $(T,)$.

For query position $i$, the allowed key set is

$$
\mathcal{A}_i = \{\, j : m_j = 1 \ \text{and}\ (\text{not causal or } j \le i) \,\},
$$

where $m_j$ is the padding mask at key position $j$. Attention probabilities are normalized
over $\mathcal{A}_i$ and are zero elsewhere.

The padding mask excludes positions as **keys and values**. It does not suppress the query
itself. A padded query can still attend to earlier valid keys and produce nonzero output. If a
query has no allowed keys after combining padding and causality, its attention probabilities and
attention output are zero. Later dense biases or residual paths can still yield nonzero final
model outputs.

The mask is not a general attention matrix and does not support arbitrary pairwise masks. It
also does not remove padded positions from losses or targets. Embedding rows are not frozen by
selecting a padding ID.

See [single-head attention masks](layers/attention.md#padding-masks) for the layer-level
contract.

## Loss masking

The built-in CCE averages over **all** output positions, including padding:

$$
L_{\mathrm{CCE}}
= -\frac{1}{R}\sum_{r=1}^{R}\sum_{j=1}^{V} y_{rj}\log p_{rj},
$$

where $R = B \cdot T$ counts padded positions. For initial experiments, equal-length unpadded
batches avoid this issue. Padding-aware training requires a corresponding masked loss and
gradient in application code.

Setting padded CCE targets to all-zero vectors removes their numerator contribution, but the
built-in denominator still counts those positions. This does not produce the mean over valid
tokens. In addition, the desired target mask for next-token prediction can differ from the input
mask after shifting targets.

A custom gradient callback receives flattened $(B \cdot T, V)$ rows. It must use the matching
flattened target-position mask, return zero gradients for ignored rows, and normalize by the
number of included target positions $n_{\text{valid}}$. Its reporting loss must use the same
reduction:

$$
L_{\text{masked}}
= -\frac{1}{n_{\text{valid}}}
\sum_{(r,j)\ \text{valid}} y_{rj}\log p_{rj}.
$$

Since this callback is not the exact built-in `gradient_cce` function, Neurality uses the
general softmax backward path. The callback must return gradients with respect to probabilities,
for example

$$
dA_{rj} =
\begin{cases}
-\dfrac{y_{rj}}{p_{rj}\,n_{\text{valid}}}, & \text{row } r \text{ valid}, \\[6pt]
0, & \text{otherwise}.
\end{cases}
$$

Do **not** supply the already-fused logits gradient $(p-y)/n_{\text{valid}}$ to that path.
See [loss masking in the loss guide](loss.md#masking-and-padding) for the callback contract.

## Generation and tokenizer persistence

Greedy generation selects the highest probability at the last position, appends that token, and
runs forward again:

$$
x_{t+1} = \arg\max_{v}\ p\!\left(v \mid x_{\le t}\right).
$$

Context must not exceed `max_length`. Truncating the context resets learned position numbering
to zero for the retained window. There is no incremental KV cache, so each call recomputes the
whole context.

The [Transformer notebook](../examples/simple_transformer/simple_transformer.ipynb) includes a
small byte-level BPE tokenizer. It starts with byte IDs, learns frequent pair merges, and
encodes the example paragraph. That tokenizer belongs to the notebook, not the library. Preserve
its vocabulary and ordered merges alongside the model; changing token IDs after training changes
the meaning of embedding rows and output classes.

A tiny model trained on one paragraph primarily demonstrates fitting that paragraph. It does not
establish generalization or language quality. Prompt text can tokenize into combinations absent
from training, even when each byte is representable.

## Related guides

- [Layer overview](layers/README.md)
- [Token embedding](layers/embedding.md)
- [Positional embedding](layers/positional-embedding.md)
- [Multi-head attention](layers/multihead-attention.md)
- [Layer normalization](layers/normalization.md)
- [Residual connections](layers/residual.md)
- [Losses and gradients](loss.md)
- [Training and evaluation](training.md)
- [Serialization](serialization.md)