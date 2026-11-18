# Residual connections

[Documentation index](../README.md) · [Layer overview](README.md) ·
[Layer normalization](normalization.md) · [Transformer guide](../transformer.md)

Residual connections let a layer or block learn a modification to its input instead of a full
replacement. Neurality implements them with two marker layers, `residual_start()` and
`residual_end()`, which add the saved input to the branch output. The markers have no trainable
parameters.

```python
model.residual_start()
# ... branch layers ...
model.residual_end()
```

Both methods return the model. Pairs may be nested.

## Forward pass

`residual_start()` pushes the current tensor onto an internal stack:

$$
r \leftarrow x.
$$

The layers between the markers form a branch function $F$. `residual_end()` pops the saved
tensor and adds it to the branch output:

$$
y = F(x) + x.
$$

The identity path carries no parameters and no nonlinearity. It requires the branch output and
the saved input to have the **same complete tensor shape**, not merely the same feature width.
A mismatch raises `ValueError("residual input and output shapes must match")`.

Because the implementation uses a stack, nested residuals work naturally:

```python
model.residual_start()          # push x1
model.norm()
model.attention(embedding_size=8)
model.residual_end()            # y1 = attention(norm(x1)) + x1

model.residual_start()          # push y1
model.norm()
model.dense(32, N.Activations.gelu)
model.dense(8, N.Activations.linear)
model.residual_end()            # y2 = ffn(norm(y1)) + y1
```

## Shapes

| Input shape | Output shape |
|---|---|
| $(T,D)$ | $(T,D)$ |
| $(B,T,D)$ | $(B,T,D)$ |

The identity path cannot change width. To use a wider feed-forward branch, expand and contract
inside the branch so that the final branch output returns to $D$:

```python
model.residual_start()
model.dense(32, N.Activations.gelu)   # 8 -> 32
model.dense(8, N.Activations.linear)  # 32 -> 8
model.residual_end()
```

`residual_end()` checks the feature width at construction time. The forward pass additionally
checks the full shape, which catches accidental sequence-length or batch changes inside the
branch.

## Backward pass

Let $dy$ be the incoming gradient of the residual output $y = F(x) + x$. The derivative splits
into two paths:

$$
\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + I.
$$

The implementation handles the split with a second stack:

- `residual_end` pushes the branch-output gradient $dy$ onto a stack.
- Layers in the branch propagate their own gradient.
- `residual_start` pops the saved gradient and adds it to the branch input gradient:

$$
dx = dF + dy.
$$

This means every residual block receives gradient flow along both the learned branch and the
identity path. No parameter gradient is produced by the markers themselves.

## Construction and validation

| Check | When it runs |
|---|---|
| `residual_end` has a matching `residual_start` | Construction |
| Feature width restored before `residual_end` | Construction |
| Pairs balanced (no unmatched start or end) | `forward()`, `save()`, `load()` |
| Complete tensor shape equal | `forward()` |

Error messages include:

- `residual_end requires a matching residual_start`
- `residual input and output feature sizes must match`
- `residual_start requires a matching residual_end`
- `unmatched residual_end`
- `residual input and output shapes must match`

## Serialization

JSON stores `residual_start` and `residual_end` layer entries with an `embedding_size` field and
no trainable parameters. Loading validates that the width matches the preceding layer at each
marker, and that the final model has balanced pairs.

## Example

The Transformer example uses pre-normalization residual blocks:

```python
model.residual_start().norm()
model.multihead_attention(num_heads=2, causal=True)
model.residual_end()

model.residual_start().norm()
model.dense(16, N.Activations.gelu, initialization=N.Initializations.glorot)
model.dense(8, N.Activations.linear, initialization=N.Initializations.glorot)
model.residual_end()
```

The first block computes

$$
y_1 = \operatorname{MHA}(\operatorname{LN}(x)) + x,
$$

and the second computes

$$
y_2 = \operatorname{FFN}(\operatorname{LN}(y_1)) + y_1.
$$

A final normalization and vocabulary projection produce token probabilities. See the
[Transformer guide](../transformer.md#a-complete-small-language-model).

## Related guides

- [Layer normalization](normalization.md)
- [Multi-head attention](multihead-attention.md)
- [Dense layer](dense.md)
- [Architecture: residual routing](../architecture.md#residual-routing)
- [Serialization](../serialization.md)