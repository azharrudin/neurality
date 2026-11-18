# Model serialization

[Documentation index](README.md) · [Architecture](architecture.md) · [API reference](api.md) ·
[Training](training.md)

Neurality saves models as plain JSON. Files contain the architecture and trained parameters,
independent of the runtime backend. They do not contain optimizer state, preprocessing,
tokenization, or experiment metadata.

## Save and load

This standalone example writes only inside a temporary directory:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from lib.neurality import Neurality as N

model = N(backend=N.Backends.PYTHON).fit([[0.0, 1.0]])
model.dense(1, N.Activations.sigmoid)

with TemporaryDirectory() as directory:
    path = Path(directory) / "model.json"
    model.save(path)
    loaded = N.load(path, backend=N.Backends.PYTHON)
    print(loaded.forward([0.0, 1.0]))
```

Behavior:

- `save()` returns the current model.
- `load()` returns a **new** instance. Always retain its result.
- `path` can be a `Path` or any filename accepted by Python's `open()`.
- Parent directories must already exist.
- Saving overwrites the chosen file and is not an atomic checkpoint operation.
- The file does not store a backend preference. Pass `backend=` explicitly when loading, or the
  default backend is used.

## Format version 2

New files use a JSON object with `format`, `version`, `input_size`, and `layers`:

```json
{
  "format": "neurality",
  "version": 2,
  "input_size": 1,
  "layers": [
    {
      "type": "dense",
      "neurons": [
        {
          "w": [0.5],
          "b": 0.0,
          "activation": "sigmoid",
          "derivative": "deriv_sigmoid"
        }
      ]
    }
  ]
}
```

Layer order determines execution order. Parameters are JSON numbers and arrays, independent of
the runtime backend.

### Layer records

| Layer `type` | Configuration fields | Parameter fields |
|---|---|---|
| `dense` | none beyond `neurons` | `neurons` with `w`, `b`, `activation`, `derivative` |
| `embedding` | `vocab_size`, `embedding_size` | `weights`, shape $(V,D)$ |
| `positional_embedding` | `embedding_size`, `max_length` | `weights`, shape $(L_{\max},D)$ |
| `norm` | `embedding_size`, `epsilon` | `gamma`, `beta`, each shape $(D,)$ |
| `attention` | `embedding_size`, `key_size`, `causal` | `Wq`, `Wk` of shape $(D,d_k)$; `Wv` of shape $(D,D)$ |
| `multihead_attention` | `embedding_size`, `num_heads`, `causal` | `Wq`, `Wk`, `Wv`, `Wo`, each shape $(D,D)$ |
| `residual_start` | `embedding_size` | none |
| `residual_end` | `embedding_size` | none |

Dense activations and derivatives are stored by name. Only the built-in pairs can be
reconstructed; see [activations](activations.md#serialization-names).

### Version 1 compatibility

The loader also accepts version 1 dense models, where each layer is a neuron list without a
typed layer wrapper. New saves always use version 2. Compatibility with future versions is not
implied.

## Validation

Saving:

1. checks that residual pairs are balanced;
2. recognizes built-in activation and derivative functions;
3. validates the generated payload through the loader;
4. serializes with non-finite JSON values disabled.

Loading validates:

- the format marker and version;
- a positive integer `input_size`;
- supported layer types;
- parameter dimensions and finite values;
- layer configuration consistency (widths, head counts, residual balance, epsilon, and boolean
  flags such as `causal`).

Malformed model structures raise

```text
ValueError: Invalid Neurality model file: ...
```

Invalid JSON syntax and file access failures come from the underlying `json` and file APIs.
Unknown activation names and custom callables cannot be reconstructed. Serialize built-in
functions directly rather than wrappers with similar behavior.

## What is omitted

| Omitted state | Application responsibility |
|---|---|
| Optimizer class, hyperparameters, moments, and steps | Reconfigure an optimizer; resumed updates start with fresh state |
| Training data and targets | Load the dataset again |
| Forward caches | Run forward before backward |
| Feature order, encoders, means, standard deviations | Store preprocessing metadata separately |
| Tokenizer vocabulary, merge order, special-token conventions | Save and restore the exact token mapping |
| Class names, thresholds, training epoch, random state | Store experiment metadata separately |

This format is a parameter checkpoint, not a complete training-session snapshot. Reloading Adam
weights without Adam moments does not reproduce uninterrupted training exactly.

## Backend interoperability and resumed training

A NumPy-saved model can be loaded with native Python and vice versa. Because the file does not
preserve a backend preference, pass it explicitly:

```python
loaded = N.load("model.json", backend=N.Backends.PYTHON)
```

See [backends](backends.md#serialization-and-portability) for the representation and parity
details behind cross-backend loading.

To continue training after loading:

1. attach an optimizer with `model.optimizer(...)`;
2. run a fresh `forward()`;
3. call `backward()` with the matched gradient function.

Calling `fit()` is unnecessary for explicit forward/backward loops, and using it to change a
loaded architecture's input width does not rebuild the existing weights. Converting an existing
model with `model.backend(...)` also resets optimizer state and removes caches.

For an experiment bundle, store the model JSON beside preprocessing/tokenizer metadata and a
description of the data split, objective, and hyperparameters. Keep the best validation
checkpoint separately from temporary saves.

## Related guides

- [Dense layer serialization](layers/dense.md)
- [Embedding serialization](layers/embedding.md)
- [Attention serialization](layers/attention.md)
- [Multi-head serialization](layers/multihead-attention.md)
- [Normalization serialization](layers/normalization.md)
- [Residual serialization](layers/residual.md)
- [API reference](api.md#serialization)