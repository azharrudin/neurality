# API reference

[Documentation index](README.md) · [Architecture](architecture.md) · [Layer overview](layers/README.md) ·
[Training](training.md)

```python
from lib.neurality import Neurality as N
```

Everything is nested inside the `Neurality` class. There is no module-level `Backends`,
`Optimizers`, `Activations`, or `backend()` function.

## Model lifecycle

| Call | Behavior and return value |
|---|---|
| `N(backend=None)` | Creates a model. Defaults to NumPy, `INPUT_SIZE = 2`, and SGD with learning rate `0.1`. |
| `model.fit(data, target=None)` | Stores data/target and sets the input feature count from the last dimension. Returns the model. Does not train. |
| `model.backend(backend)` | Converts data and parameters to the selected backend, resets optimizer state, and clears forward caches. Returns the model. |
| `model.optimizer(optimizer)` | Copies optimizer configuration and starts with fresh optimizer state. Returns the model. |
| `model.forward(data, padding_mask=None, training=None)` | Returns predictions and replaces all forward caches. `training` overrides the model mode for this call. |
| `model.backward(target, gradient_function, learning_rate=None)` | Computes gradients and immediately applies one optimizer update. Returns the model. |
| `model.train(mode=True)` | Sets training mode. Dropout is active only in training mode. Returns the model. |
| `model.eval()` | Equivalent to `train(False)`. Returns the model. |
| `model.save(path)` | Writes JSON and returns the model. |
| `N.load(path, backend=None)` | Class method returning a new model. |

### Backend selection

```python
model = N(backend=N.Backends.PYTHON)
model = N(backend="python")        # enum value also accepted
model.backend(N.Backends.NUMPY)
```

See [backends](backends.md) for the native-Python and NumPy execution models, runtime
conversion rules, and numerical differences.

`N.backends` is an alias of `N.Backends`. The constructor and instance method accept the enum
or its string values `"python"` and `"numpy"`.

Select the backend explicitly at construction when NumPy may be unavailable. `N()` tries to
import NumPy immediately and raises `ImportError` if it is missing; a later
`model.backend(N.Backends.PYTHON)` cannot run.

`model.backend(...)` resets optimizer state and clears the forward cache even when reselecting
the same backend. Avoid it inside a training loop.

### `fit`

```python
model.fit(data, target=None)
```

- `data` must be a non-empty, finite matrix of rank 2 or 3.
- `INPUT_SIZE` is set from `data.shape[-1]`.
- For dense models, call `fit()` before adding layers when the input width is not the default
  of 2.
- Calling it later does not rebuild existing weights if the input width changes.
- Token models normally skip it and start with `embedding(vocab_size, embedding_size)`.

## Shape conventions

| Pipeline | Input | Output |
|---|---|---|
| Dense, one sample | $(D,)$ | $(1,C)$ |
| Dense batch | $(B,D)$ | $(B,C)$ |
| Dense over sequences | $(B,T,D)$ | $(B,T,C)$ |
| Token embedding | $(T,)$ or $(B,T)$ | $(1,T,D)$ or $(B,T,D)$ |
| Standalone attention | $(T,D)$ or $(B,T,D)$ | same shape |
| Positional embedding, norm, dropout, residual | feature matrix or batch | same shape |

A 2D matrix passed to attention represents **one sequence**; its rows attend to one another. A
3D tensor represents independent sequences.

Inputs must be rectangular, non-empty, finite, and match the model feature width (or contain
valid token IDs for embedding models).

## Layer construction

All builders return the model and support chaining.

| Method | Summary | Details |
|---|---|---|
| `dense(number, activation_function, derivative=None, bias=0, initialization=None)` | Fully connected layer | [Dense](layers/dense.md) |
| `embedding(vocab_size, embedding_size)` | Trainable token table; must be first | [Embedding](layers/embedding.md) |
| `positional_embedding(max_length)` | Learned position vectors | [Positional](layers/positional-embedding.md) |
| `norm(epsilon=1e-5)` | Layer normalization over the final feature axis | [Normalization](layers/normalization.md) |
| `dropout(rate=0.1)` | Training-time elementwise dropout with inverted scaling | [Dropout](layers/dropout.md) |
| `residual_start()` | Push the current tensor for a skip connection | [Residual](layers/residual.md) |
| `residual_end()` | Add the saved tensor to the branch output | [Residual](layers/residual.md) |
| `attention(key_size=None, causal=False, embedding_size=None)` | Single-head self-attention | [Attention](layers/attention.md) |
| `multihead_attention(num_heads, causal=False, embedding_size=None)` | Multi-head self-attention with output projection | [Multi-head](layers/multihead-attention.md) |

Key defaults and constraints:

- **Dense.** `number` must be a positive integer. `activation_function` is required. The default
  derivative is `N.Derivations.deriv_<activation name>`. `bias` must be finite. A custom
  initializer receives `(fan_in, fan_out)` and returns one finite vector of length `fan_in`.
- **Embedding.** Both sizes must be positive integers. Input IDs must be integral and in
  $[0,V)$. There is no reserved padding ID or tokenizer.
- **Positional embedding.** `max_length` must be a positive integer. The feature width follows
  the preceding layer or `INPUT_SIZE`. There is no embedding-size argument.
- **Norm.** `epsilon` must be positive, finite, and not a boolean.
- **Dropout.** `rate` must be finite and in $[0,1)$; booleans are rejected. The layer has no
  trainable parameters. Dropout is active only in training mode.
- **Residual markers.** Pairs must be balanced and restore the feature width. Forward also
  requires the complete tensor shape to match.
- **Attention.** `key_size` defaults to the feature width. `causal` must be a real boolean.
  `embedding_size` sets `INPUT_SIZE` for a first layer and must match the preceding width
  otherwise. Single-head attention has no output projection.
- **Multi-head attention.** `num_heads` must be positive and divide the feature width. All four
  projection matrices have shape $(D,D)$. Output width equals input width.

## Forward masks

`padding_mask` uses `1`/`True` for valid tokens and `0`/`False` for padding. Shape is $(B,T)$,
or $(T,)$ for a single sequence. It applies to every attention layer in the model.

The mask excludes padded positions as **keys and values**. It does not exclude padded queries,
does not freeze embedding rows, and does not remove padded positions from losses. It is not a
general pairwise attention mask. See
[attention masks](layers/attention.md#padding-masks) and
[Transformer mask semantics](transformer.md#mask-semantics).

## Training mode

New models start in training mode. `model.eval()` disables dropout for validation and
generation; `model.train()` re-enables it. Passing `training=False` to `forward()` disables
dropout for one call without changing the model mode. Other layers behave identically in both
modes, and there are no running statistics to update.

## Backward callback

`backward()` requires a preceding `forward()`. The target must contain the same number of
elements as the output; the implementation reshapes it to the output shape. Prefer an exact
shape match.

A custom `gradient_function(prediction_rows, target_rows)` receives flattened rows of shape
$(R,C)$ and must return that shape. It represents

$$
\frac{\partial L}{\partial p},
$$

the gradient with respect to model output, not logits. Built-in gradients include averaging.
A positive, finite `learning_rate` override applies only to the current update.

For the exact built-in `gradient_cce` on a final softmax layer, the implementation uses a fused
logit-gradient shortcut. Any wrapper uses the general probability-gradient path. See
[losses](loss.md#fused-softmax-and-cce).

## Losses

| Loss | Matching gradient | Reduction |
|---|---|---|
| `N.Loss.loss_mse(prediction, target)` | `N.Gradients.gradient_mse` | Mean over all elements |
| `N.Loss.loss_bce(prediction, target)` | `N.Gradients.gradient_bce` | Mean over all elements |
| `N.Loss.loss_cce(prediction, target)` | `N.Gradients.gradient_cce` | Sum over classes, mean over rows |

Losses return Python scalars. CCE expects distribution or one-hot targets, not sparse class IDs.
Built-in gradients are already normalized. See [losses and gradients](loss.md).

## Optimizers

| Constructor | Defaults | Constraints |
|---|---|---|
| `N.Optimizers.SGD(learning_rate=0.01)` | $0.01$ | Rate positive and finite |
| `N.Optimizers.Momentum(learning_rate=0.01, momentum=0.9)` | $0.01$, $0.9$ | Momentum in $[0,1)$ |
| `N.Optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8)` | $0.001$, $0.9$, $10^{-8}$ | Rho in $[0,1)$; epsilon positive and finite |
| `N.Optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)` | $0.001$, $0.9$, $0.999$, $10^{-8}$ | Betas in $[0,1)$; epsilon positive and finite |

The implicit model optimizer uses SGD at $0.1$, whereas `SGD()` defaults to $0.01$. Attaching
an optimizer resets its accumulated state. See [optimizers](optimizers.md).

## Math helpers and inspection

| Helper | Use |
|---|---|
| `N.Math.shape(value)` | Shape tuple for lists or arrays |
| `N.Math.tolist(value)` | Convert nested values to Python lists/scalars |
| `N.Math.flatten(value)` | Flat list of scalars |
| `N.Math.array(value, backend)` | Copy into the selected representation |
| `N.Math.reshape(value, shape)` | Reshape with element-count validation |
| `N.Math.rows(value)` | Flatten leading dimensions into rows |
| `N.Math.transpose(value)` | Matrix transpose |
| `N.Math.matmul(a, b)` | Matrix multiplication |

The adapter is not a complete NumPy replacement. NumPy paths may broadcast operands where pure
Python requires equal list shapes, so supply exact target shapes. `N.Math.randn()` follows the
internal global backend default; built-in initializers passed to `dense()` use the model's
backend.

## Serialization

| Call | Behavior |
|---|---|
| `model.save(path)` | Validates residuals and activations, round-trips the payload through the loader, writes JSON |
| `N.load(path, backend=None)` | Returns a new model; the file does not store a backend preference |

Model files contain architecture and parameters only. Optimizer state, training data,
preprocessing, tokenizers, and class labels are not included. See
[serialization](serialization.md).

## Exceptions

| Situation | Exception |
|---|---|
| Wrong argument type | `TypeError` |
| Invalid size, shape, value, configuration | `ValueError` |
| Missing NumPy for the NumPy backend | `ImportError` |
| Missing forward cache | `ValueError("Call forward before backward")` |
| Malformed model file | `ValueError("Invalid Neurality model file: ...")` |
| Invalid JSON syntax or file access | Underlying `json`/file exception |

## Related guides

- [Architecture and execution](architecture.md)
- [Training and evaluation](training.md)
- [Examples and troubleshooting](examples-and-troubleshooting.md)