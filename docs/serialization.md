# Model serialization

[Documentation index](README.md)

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

`save()` returns the current model. `load()` returns a new instance: always retain its result. A `Path` or filename accepted by Python `open()` can be used. Parent directories must already exist. Saving overwrites the chosen file and is not an atomic checkpoint operation.

## Format version 2

New files use a JSON object containing `format`, `version`, `input_size`, and `layers`. This minimal example is a valid one-input, one-output dense model:

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

Layer order determines execution order. Weights are JSON numbers/arrays, independent of the runtime backend.

| Layer type | Configuration and parameter fields |
|---|---|
| `dense` | `neurons`, each with `w`, `b`, `activation`, `derivative` |
| `embedding` | `vocab_size`, `embedding_size`, `weights` of shape `(V,D)` |
| `positional_embedding` | `embedding_size`, `max_length`, `weights` of shape `(max_length,D)` |
| `norm` | `embedding_size`, `epsilon`, vectors `gamma`, `beta` |
| `attention` | `embedding_size`, `key_size`, `causal`, matrices `Wq`, `Wk`, `Wv` |
| `multihead_attention` | `embedding_size`, `num_heads`, `causal`, matrices `Wq`, `Wk`, `Wv`, `Wo` |
| `residual_start` / `residual_end` | `embedding_size`; no trainable parameters |

Single-head `Wq` and `Wk` have shape `(D,key_size)` and `Wv` has shape `(D,D)`. All multi-head projection matrices have shape `(D,D)`.

The loader also accepts version 1 dense models, where each layer is a neuron list without a typed layer wrapper. New saves always use version 2. Compatibility with future versions is not implied.

## Validation

Saving checks that residual pairs are balanced, recognizes built-in activation/derivative functions, validates the generated payload through the loader, and serializes with nonfinite JSON values disabled. Loading validates the format/version, positive input width, supported layer types, parameter dimensions and finite values, and layer configuration consistency.

Malformed model structures raise `ValueError` with an `Invalid Neurality model file` prefix. Invalid JSON syntax and file access failures come from the underlying JSON/file APIs. Unknown activation names and custom callables cannot be reconstructed. Serialize built-in functions directly rather than wrappers with similar behavior.

## What is omitted

| Omitted state | Application responsibility |
|---|---|
| Optimizer class, hyperparameters, moments, and steps | Reconfigure an optimizer; resumed updates start with fresh state |
| Training data and targets | Load the dataset again |
| Forward caches | Run forward before backward |
| Feature order, encoders, means, standard deviations | Store preprocessing metadata separately |
| Tokenizer vocabulary, merge order, special token conventions | Save and restore the exact token mapping |
| Class names, thresholds, training epoch, random state | Store experiment metadata separately |

This format is a model-parameter checkpoint, not a complete training-session snapshot. Reloading Adam weights without Adam moments does not reproduce uninterrupted training exactly.

## Backend interoperability and resumed training

A NumPy-saved model can be loaded with pure Python and vice versa. Pass `backend` explicitly to `N.load()`; otherwise the default backend is used, which requires NumPy unless configured internally otherwise. The file does not preserve a backend preference.

To continue training after loading, set an optimizer once, run a fresh forward, and call backward with a matching gradient. Calling `fit()` is unnecessary for explicit forward/backward loops and should not be used to change a loaded architecture's input width. Converting an existing model with `model.backend(...)` also resets optimizer state and removes caches.

For an experiment bundle, store the model JSON beside preprocessing/tokenizer metadata and a description of the data split, objective, and hyperparameters. Keep the best validation checkpoint separately from temporary saves.
