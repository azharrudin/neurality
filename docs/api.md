# API reference

[Documentation index](README.md)

```python
from lib.neurality import Neurality as N
```

## Model lifecycle

| Call | Behavior and return value |
|---|---|
| `N(backend=None)` | Creates a model; defaults to NumPy, two input features, and SGD with learning rate `0.1`. |
| `model.fit(data, target=None)` | Stores data/target and sets the input feature count from the last dimension. Returns the model; does not train. |
| `model.backend(backend)` | Converts data and parameters, resets optimizer state, clears caches. Returns the model. |
| `model.optimizer(optimizer)` | Copies optimizer configuration and starts with fresh optimizer state. Returns the model. |
| `model.forward(data, padding_mask=None)` | Returns predictions and replaces forward caches. |
| `model.backward(target, gradient_function, learning_rate=None)` | Computes gradients and updates parameters. Returns the model. |
| `model.save(path)` | Writes JSON and returns the model. |
| `N.load(path, backend=None)` | Class method returning a new model. |

Use `N.Backends.PYTHON` or `N.Backends.NUMPY`; `N.backends` is an alias of that enum. Constructor and instance backend selection also accept enum values `"python"` and `"numpy"`. These classes are nested in `Neurality`; there is no module-level `neurality.Backends`, `neurality.Optimizers`, or `neurality.backend()` API.

Select the backend explicitly at construction, especially without NumPy. Calling `N()` first tries the default NumPy backend before a later `model.backend(...)` can run.

`fit()` requires nonempty, finite data of rank two or three. Call it before adding dense layers: it does not rebuild existing weights if the input feature count changes. Token models normally start directly with `embedding()`.

## Shape conventions

| Pipeline | Input | Output |
|---|---|---|
| Dense | `(D,)` or `(B,D)` | `(1,C)` or `(B,C)` |
| Dense over sequences | `(B,T,D)` | `(B,T,C)` |
| Token embedding | `(T,)` or `(B,T)` | `(1,T,D)` or `(B,T,D)` |
| Standalone attention | `(T,D)` or `(B,T,D)` | Same shape |
| Positional embedding / norm | Feature matrix or batch of matrices | Same shape |

A 2D matrix passed into attention represents one sequence. Rows can attend to each other. Use 3D input to separate sequences. Inputs must be rectangular, nonempty, finite, and match the model feature count.

## Layer construction

All layer-building methods return the model and support chaining.

### `dense(number, activation_function, derivative=None, bias=0, initialization=None)`

`number` is a positive integer. `activation_function` is required. The default derivative is looked up as `deriv_<activation name>` in `Derivations`. Each neuron receives the same finite initial bias.

| Built-in activation | Derivative |
|---|---|
| `N.Activations.relu` | `N.Derivations.deriv_relu` |
| `N.Activations.leaky_relu` | `N.Derivations.deriv_leaky_relu` |
| `N.Activations.gelu` | `N.Derivations.deriv_gelu` |
| `N.Activations.sigmoid` | `N.Derivations.deriv_sigmoid` |
| `N.Activations.tanh` | `N.Derivations.deriv_tanh` |
| `N.Activations.linear` | `N.Derivations.deriv_linear` |
| `N.Activations.softmax` | `N.Derivations.deriv_softmax` |

Derivatives receive preactivation values. Direct `deriv_softmax()` calls return a Jacobian per row; dense backward handles the built-in softmax specially. Direct `Activations.softmax()` requires a 2D matrix, while dense layers flatten and restore sequence dimensions automatically.

| Initialization | Per-neuron weight distribution |
|---|---|
| `None` | Normal with standard deviation `0.1` |
| `N.Initializations.kaiming` | Normal with standard deviation `sqrt(2/fan_in)` |
| `N.Initializations.glorot` | Normal with standard deviation `sqrt(2/(fan_in+fan_out))` |

A custom initializer receives `(fan_in, fan_out)` and must return one finite vector of length `fan_in`, rather than the full layer matrix. Custom activations require an explicit derivative unless the naming lookup resolves one. JSON saving supports only built-in activations and derivatives.

### `embedding(vocab_size, embedding_size)`

Must be the first layer. Both sizes must be positive integers. Input values must be integral token IDs in `[0,vocab_size)`. The trainable table has shape `(V,D)`. There is no reserved padding ID or built-in tokenizer.

### `positional_embedding(max_length)`

Adds learned position vectors. `max_length` must be a positive integer. Feature width follows the preceding layer, or the model input width when this is the first layer. Sequences longer than the capacity are rejected. There is no `embedding_size` parameter on this method.

### `norm(epsilon=1e-5)`

Applies layer normalization over the final dimension. `epsilon` must be positive and finite, and cannot be boolean. Trainable `gamma` and `beta` have shape `(D,)`.

### `residual_start()` / `residual_end()`

Open and close a skip connection. Closing requires a matching start and equal feature widths. Forward also checks the complete tensor shape. Unclosed residual regions are rejected by forward and save. Pairs may be nested.

### `attention(key_size=None, causal=False, embedding_size=None)`

Adds single-head self-attention. `key_size` must be positive and defaults to the feature width. `causal` must be boolean. Set `embedding_size` for a first layer whose input width differs from the default; after another layer it must match the preceding output width. Output width equals input width.

### `multihead_attention(num_heads, causal=False, embedding_size=None)`

Adds multi-head self-attention and output projection. `num_heads` must be a positive integer dividing the feature width. Each head uses `D/num_heads` features. The other arguments follow single-head attention rules.

## Forward masks and backward callbacks

`padding_mask` uses `1`/`True` for valid tokens and `0`/`False` for padding. Shape is `(B,T)`, or `(T,)` for a single sequence. It applies to every attention layer. It does not exclude padded queries or targets from the loss. See [masking details](transformer.md#mask-semantics).

`backward()` requires a preceding forward. Prefer a target with the exact output shape. The implementation can reshape a target with the same number of elements, but loss calls and broadcasting may behave differently if the original shape is wrong.

A custom `gradient_function(prediction_rows, target_rows)` receives arrays/lists of shape `(R,C)` after flattening leading dimensions and must return that shape. It represents a derivative with respect to model output, not logits. Built-in gradients include averaging. A positive, finite `learning_rate` override affects only the current update.

## Losses

| Loss function | Matching gradient |
|---|---|
| `N.Loss.loss_mse(prediction, target)` | `N.Gradients.gradient_mse` |
| `N.Loss.loss_bce(prediction, target)` | `N.Gradients.gradient_bce` |
| `N.Loss.loss_cce(prediction, target)` | `N.Gradients.gradient_cce` |

Losses return Python scalars. MSE and BCE average over all elements; CCE sums over classes and averages over sample/token rows. CCE expects distributions or one-hot targets, not sparse class IDs.

## Optimizers

| Constructor | Defaults and constraints |
|---|---|
| `N.Optimizers.SGD(learning_rate=0.01)` | Positive, finite rate |
| `N.Optimizers.Momentum(learning_rate=0.01, momentum=0.9)` | Momentum in `[0,1)` |
| `N.Optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8)` | Rho in `[0,1)`; positive, finite epsilon |
| `N.Optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)` | Betas in `[0,1)`; positive, finite epsilon |

The implicit model optimizer uses SGD `0.1`, whereas an explicitly constructed `SGD()` defaults to `0.01`. Set the optimizer once before training. Repeated `model.optimizer(...)` calls reset accumulated moments and step counts.

## Math helpers and inspection

`N.Math.shape(value)`, `N.Math.tolist(value)`, and `N.Math.flatten(value)` inspect results across backends. `N.Math.array(value, backend)` copies a value into the selected representation. Helpers such as `reshape`, `rows`, `transpose`, and `matmul` support internal tensor operations.

This adapter is not a complete NumPy replacement. In particular, NumPy may broadcast operands where pure Python requires equal list shapes. Supply targets with exact shapes. Direct `Math.randn()` and initializer calls follow the internal global backend default; built-in initializers passed to `dense()` use the model instance backend.
