# Layer guides

[Documentation index](../README.md) · [Architecture](../architecture.md) · [API reference](../api.md)

Every layer is added with a builder method that returns the model, so calls can be chained:

```python
model = N(backend=N.Backends.PYTHON).fit(X, y)
model.dense(16, N.Activations.gelu).dense(1, N.Activations.sigmoid)
```

Layers execute in insertion order. There is no separate compile step, no automatic shape
inference graph, and no branching syntax beyond residual start/end markers.

## Layer catalog

| Guide | JSON `type` | Forward map | Last-dim output | Trainable parameters |
|---|---|---|---|---|
| [Dense](dense.md) | `dense` | $f(XW + \mathbf{1}b^\top)$ | $C$ | One $w$ and $b$ per neuron |
| [Embedding](embedding.md) | `embedding` | $E[x]$ | $D$ | Table $E \in \mathbb{R}^{V \times D}$ |
| [Positional embedding](positional-embedding.md) | `positional_embedding` | $h_t + P_t$ | unchanged | Table $P \in \mathbb{R}^{L_{\max} \times D}$ |
| [Attention](attention.md) | `attention` | $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ | unchanged | $W_Q, W_K, W_V$ |
| [Multi-head attention](multihead-attention.md) | `multihead_attention` | $\operatorname{Concat}(O_1,\dots,O_H)W_O$ | unchanged | $W_Q, W_K, W_V, W_O$ |
| [Layer normalization](normalization.md) | `norm` | $\gamma \odot \hat{x} + \beta$ | unchanged | $\gamma, \beta$ |
| [Residual start](residual.md) | `residual_start` | pushes $x$ onto a stack | unchanged | none |
| [Residual end](residual.md) | `residual_end` | $F(x) + x$ | unchanged | none |

The forward maps above assume an input matrix $X \in \mathbb{R}^{R \times D}$ or a sequence
tensor with final feature width $D$. Each guide defines all symbols.

## Common conventions

### Shapes

A model input is one of:

| Input shape | Meaning | Typical use |
|---|---|---|
| $(D,)$ | One sample | Single dense prediction |
| $(B,D)$ | Batch of samples | Dense models |
| $(T,)$ | One token sequence | Embedding input (promoted to $(1,T)$) |
| $(B,T)$ | Batch of token sequences | Embedding input |
| $(T,D)$ | One feature sequence | Standalone attention |
| $(B,T,D)$ | Batch of feature sequences | Attention, normalization, residuals |

A batch must be rectangular: every sequence has the same length. Padding belongs in
application code and is communicated to attention through `padding_mask`.

### Building order

- `fit(data, target=None)` sets the input width for a dense model. Call it before the first
  dense layer when the width is not the default of 2. It stores the data and records the
  last-dimension width; it does not train.
- Token models normally begin with `embedding(...)` and skip `fit()`. The embedding declares its
  own vocabulary and width.
- `positional_embedding`, `norm`, `residual_start`, and attention can be first layers. A first
  layer takes its width from `INPUT_SIZE`, which defaults to 2 unless `fit()` or an
  `embedding_size` argument sets it.
- Changing `INPUT_SIZE` after building dense layers does not rebuild existing weights.

### Activation and derivative conventions

Dense layers store both an activation callable $f$ and a derivative callable $f'$. The
derivative receives **preactivation** values $Z$, not post-activation values $A$. The default
derivative is looked up by name:

```python
derivative = getattr(N.Derivations, f"deriv_{activation_function.__name__}")
```

Built-in activations resolve automatically. A custom activation must be paired with an explicit
derivative, and it cannot be serialized. See [activations](../activations.md).

### Masks

`forward(data, padding_mask=...)` accepts a padding mask for every attention layer in the model.
The mask marks valid tokens with `1`/`True` and padding with `0`/`False`. Attention uses it to
exclude padded **keys and values**; it does not exclude padded queries, and it does not mask
losses. See [attention masks](attention.md#padding-masks) and
[transformer mask semantics](../transformer.md#mask-semantics).

### Residual pairing

`residual_start()` and `residual_end()` must be balanced, and the feature width must be
restored before the matching end marker. Pairs may be nested. Forward, save, and load all
validate pairing. See [residual connections](residual.md).

### Serialization

Every layer type has a JSON representation. Built-in activations and derivatives are stored by
name; parameters are stored as nested JSON numbers, independent of the runtime backend. See
[serialization](../serialization.md).

## Choosing layers

- **Tabular or vector input:** one or more [dense](dense.md) layers.
- **Token input:** [embedding](embedding.md) → optional [positional embedding](positional-embedding.md)
  → attention block(s) → output projection.
- **Transformer-style block:** pre-normalize, attend, residual-add, pre-normalize, feed-forward,
  residual-add. Neurality provides the pieces but no packaged block; see
  [Attention and Transformers](../transformer.md).
- **Classification head:** dense with softmax for mutually exclusive classes, or sigmoid for
  independent binary labels. See [losses](../loss.md) for the matching objective.