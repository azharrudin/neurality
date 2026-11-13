# Examples, testing, and troubleshooting

[Documentation index](README.md)

## Running notebooks

The library's pure Python backend needs only the standard library. Notebook tooling and preprocessing have additional dependencies. From the repository root:

```sh
python -m pip install numpy pandas scikit-learn matplotlib jupyterlab
cd examples/titanic_survivor
python -m jupyterlab
```

The Titanic notebooks import from `../../lib` and read local filenames. Their working directory must be `examples/titanic_survivor`. For the Transformer notebook, launch Jupyter from `examples/simple_transformer`; its library dependency is NumPy for the default backend, plus Jupyter tooling to run the notebook.

### Titanic classification

[training.ipynb](../examples/titanic_survivor/training.ipynb) reads `Titanic-Dataset.csv`, fills missing values, one-hot encodes `Embarked`, standardizes features, builds a dense classifier, trains with BCE, and writes `model.json`. Its layers use tanh, ReLU, and sigmoid.

The notebook's reported accuracy is computed on its training data. It is a fitting demonstration, not a held-out performance estimate. For evaluation, split data first and fit imputation, encoding, and scaling only on the training split. Save feature column order, categorical mapping, mean, and standard deviation separately from Neurality's JSON model.

[inference.ipynb](../examples/titanic_survivor/inference.ipynb) loads the saved model and passes a preprocessed seven-feature vector. Raw passenger fields cannot be substituted directly. The vector must use the same feature order and transformations as training. The supplied JSON is an example artifact, not a guarantee of model quality.

The training notebook calls NumPy methods such as `.ravel()` on predictions. If adapting it to pure Python, use `np.asarray(prediction).ravel()` or operate on the nested lists directly. `OneHotEncoder(sparse_output=False)` requires a scikit-learn installation exposing that argument.

### Simple Transformer

[simple_transformer.ipynb](../examples/simple_transformer/simple_transformer.ipynb) implements a byte-level BPE tokenizer, trains a small causal model on a single paragraph, and generates tokens greedily. Its model uses learned token/position embeddings, multi-head attention, pre-normalization residual blocks, GELU, and vocabulary softmax.

The tokenizer is notebook code, not a Neurality API. Keep learned merges and vocabulary with any saved model. A one-paragraph fitting exercise does not demonstrate broad language capabilities. Long sequence lengths increase attention work quadratically; start with the short synthetic example in the [Transformer guide](transformer.md).

## Running the tests

From the repository root, with NumPy installed:

```sh
python -m unittest discover -s tests -v
```

| Test module | Main coverage |
|---|---|
| `test_attention.py` | Attention gradients, causal behavior, token learning, backend/optimizer parity, JSON, pure Python operation without NumPy |
| `test_gelu.py` | GELU values, finite-difference derivative, dense backward, serialization |
| `test_multihead_attention.py` | Multi-head parameter gradients, padding/causal masks, validation, save/load |
| `test_transformer_components.py` | Position/norm/residual gradients, nested residual validation, backend round trips |

Finite-difference checks compare analytical updates with loss changes under small parameter perturbations. Round-trip tests compare predictions with numerical tolerance. These tests verify implementation behavior; they are not performance benchmarks or evidence of convergence for arbitrary datasets.

## Common problems

| Symptom | Cause and remedy |
|---|---|
| `ModuleNotFoundError: lib` | Run standalone examples from the repository root. For notebooks, use the documented working directory and import path. |
| NumPy import error during `N()` | NumPy is the default backend. Install it or construct `N(backend=N.Backends.PYTHON)` directly. |
| Missing `neurality.Backends` or `neurality.Optimizers` | Import the class as `N` and use `N.Backends` / `N.Optimizers`; these are nested classes. |
| `Call forward before backward` | Run forward first; backend conversion also clears the required cache. |
| Incorrect training after validation | Validation replaced the training cache. Put validation after the update, or rerun the training forward. |
| Input-shape or matrix multiplication error | Call `fit()` before dense construction, preserve input width, and distinguish `(B,D)` from `(B,T,D)`. |
| Target/gradient shape error | Match output shape; CCE needs one-hot/distribution targets. Custom callbacks receive flattened rows. |
| Inconsistent results across backends | Check exact target shapes and NumPy broadcasting; compare identical parameters with floating-point tolerance. |
| Invalid token ID | Use integral IDs in `[0,V)`, with the same tokenizer mapping used during training. |
| Invalid head count | Choose a positive head count that divides the feature width. |
| Sequence exceeds positional capacity | Shorten/window the sequence or construct and train a model with a larger capacity. |
| Residual shape or pairing error | Close each start and restore the original feature width before the end. |
| Padding still changes loss | Attention masks do not mask targets. Use equal-length unpadded batches or an application-defined masked loss and gradient. |
| Optimizer appears to forget momentum | Set it once; replacing the optimizer, changing backend, and loading a model all reset state. |
| Saving rejects an activation | Only built-in activation and derivative functions are serializable. |
| Predictions differ after reload in application | Check preprocessing, tokenizer, and backend tolerance. Those mappings are not embedded in JSON. |
| Loss becomes unstable | Check finite input, scaling, objective pairing, target encoding, and learning rate. |

## Scope and extension points

Neurality is an educational implementation. It has no GPU execution, convolution/recurrent layers, general autodiff graph, built-in trainer, early stopping, tokenizer, dropout, or integrated masked-loss API. Pure Python support does not establish MicroPython or microcontroller compatibility.

For custom experiments, use explicit dense activations/derivatives, custom initializers, or probability-gradient callbacks through the existing APIs. Adding a new trainable layer requires coordinated changes to construction, forward, backward, parameter conversion, serialization/loading, and validation. Add numerical gradient and backend round-trip checks for substantive layer changes.
