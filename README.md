# Neurality

![Neurality logo](assets/logo.png)

**Neurality** is an educational deep learning library contained in a single Python file,
[`lib/neurality.py`](lib/neurality.py). It exposes forward propagation, backpropagation, and
parameter updates for dense neural networks and Transformer-style sequence models, with
interchangeable pure-Python and NumPy backends.

The design goal is transparency rather than raw performance. There is no automatic
differentiation, no hidden trainer, and no separate training abstraction. You assemble the
layers, run `forward()` yourself, choose the loss and its gradient, and call `backward()` to
update parameters with the selected optimizer.

## Table of contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart: binary classification](#quickstart-binary-classification)
- [How training works](#how-training-works)
- [Mathematics at a glance](#mathematics-at-a-glance)
- [Documentation](#documentation)
- [Repository layout](#repository-layout)
- [Examples](#examples)
- [Running the tests](#running-the-tests)
- [Design notes and limitations](#design-notes-and-limitations)

## Highlights

| Area | What is available |
|---|---|
| Backends | Pure Python (standard library only) or NumPy, selected per model |
| Dense networks | Stacked dense layers with ReLU, Leaky ReLU, GELU, sigmoid, tanh, linear, and softmax activations |
| Sequences | Trainable token embeddings, learned positional embeddings, causal masks, and padding masks |
| Attention | Single-head self-attention and multi-head self-attention with an output projection |
| Normalization | Layer normalization over the final feature dimension |
| Residuals | Nested skip connections with automatic gradient routing |
| Objectives | Mean squared error, binary cross-entropy, and categorical cross-entropy, each with a matching gradient |
| Optimizers | SGD, Momentum, RMSprop, and Adam |
| Initialization | Normal, Kaiming-style, and Glorot-style weight initialization |
| Persistence | Versioned JSON save/load that works across both backends |

Training loops, batching, evaluation, preprocessing, tokenization, masked objectives, and
generation are intentionally left to application code.

## Requirements

- **Python 3.8 or newer.** The implementation uses `math.prod`, f-strings, and standard
  library enums.
- **NumPy is optional.** It is required for the default backend and for the full test suite.
  The pure-Python backend uses only the standard library.

## Installation

Neurality is not published as a package. Clone the repository and run code from its root, or
add the root to `PYTHONPATH`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

Imports use the repository layout directly:

```python
from lib.neurality import Neurality as N
```

All standalone examples in the documentation are written to run from the repository root.
Notebooks have their own working-directory requirements, described in the
[examples guide](docs/examples-and-troubleshooting.md).

## Quickstart: binary classification

The following example learns OR with a small dense network. It uses the pure-Python backend, so
it runs without NumPy installed.

```python
import random

from lib.neurality import Neurality as N

random.seed(42)
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [1.0]]

# fit() stores the data and records that the input has two features.
# It does not train the model.
model = N(backend=N.Backends.PYTHON).fit(X, y)

model.dense(4, N.Activations.tanh, initialization=N.Initializations.glorot)
model.dense(1, N.Activations.sigmoid, initialization=N.Initializations.glorot)
model.optimizer(N.Optimizers.Adam(learning_rate=0.03))

for epoch in range(500):
    prediction = model.forward(X)
    loss = N.Loss.loss_bce(prediction, y)
    model.backward(y, N.Gradients.gradient_bce)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: loss={loss:.6f}")

print([int(row[0] >= 0.5) for row in model.forward(X)])
```

Expected output:

```text
Epoch 100: loss=0.006248
Epoch 200: loss=0.002106
Epoch 300: loss=0.001114
Epoch 400: loss=0.000704
Epoch 500: loss=0.000491
[0, 1, 1, 1]
```

The final prediction runs one more `forward()` after the last update, so the printed labels
come from the trained parameters. Calling `forward()` without a following `backward()` is safe:
it only replaces the cached activations.

### Save and reload

Model files are plain JSON:

```python
model.save("model.json")
loaded = N.load("model.json", backend=N.Backends.PYTHON)

probability = float(loaded.forward([0.0, 1.0])[0][0])
print(probability, int(probability >= 0.5))
```

JSON stores the architecture and parameters. It does **not** store optimizer state,
preprocessing metadata, tokenizer mappings, or class labels. See
[serialization](docs/serialization.md).

## How training works

Neurality exposes a four-step loop, and every parameter update follows it:

```text
forward(batch)  ->  reporting loss  ->  backward(target, matching gradient)  ->  next batch
```

1. **Build.** Add layers in execution order. Each builder returns the model, so calls can be
   chained. For dense models, call `fit()` first so the input width is known.
2. **Forward.** `forward(data)` computes the output and replaces the internal cache. It returns
   the prediction but does not compute a loss.
3. **Backward.** `backward(target, gradient_function)` uses the most recent forward cache to
   compute parameter gradients, then immediately applies one optimizer update. It is not a
   gradient-accumulation API.
4. **Report.** Loss functions under `N.Loss` are separate from training. Compute the reporting
   loss before `backward()` when you want the loss for the parameters that produced the
   prediction.

Keep each training forward next to its backward. Running validation between them replaces the
cache, so the update would use the wrong activations. Built-in gradient functions already
include their averaging factors; do not divide them by the batch size again.

## Mathematics at a glance

For a batch matrix $X \in \mathbb{R}^{R \times D}$, a dense layer computes

$$
Z = XW + \mathbf{1}b^\top, \qquad A = f(Z),
$$

where $W \in \mathbb{R}^{D \times C}$ stacks one weight vector per output neuron, $b \in
\mathbb{R}^{C}$ is the bias, and $f$ is the activation function.

Single-head self-attention with key width $d_k$ computes

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,
\qquad
\mathrm{Attention}(X) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

The training objective is minimized by gradient descent. Every optimizer applies an update of
the form

$$
\theta \leftarrow \theta - \eta\, u(g),
$$

where $g = \partial L / \partial \theta$, $\eta$ is the learning rate, and $u$ is the
optimizer-specific direction (the raw gradient for SGD, a momentum-smoothed direction for
Momentum, a normalized direction for RMSprop and Adam).

Each guide derives the layer, loss, and optimizer equations used by the implementation, including
the backward pass.

## Documentation

Start with the [documentation index](docs/README.md).

| Guide | Contents |
|---|---|
| [Architecture](docs/architecture.md) | Internal representation, forward/backward mechanics, tensor shapes, caches, backends |
| [Backends](docs/backends.md) | NumPy vs native Python: selection, representation, runtime switching, parity, performance |
| [Layer guides](docs/layers/README.md) | One page per layer: dense, embeddings, attention, normalization, residuals |
| [Dense layer](docs/layers/dense.md) | Construction, forward equation, weight layout, softmax handling, gradients |
| [Token embeddings](docs/layers/embedding.md) | Lookup tables, index validation, gradient accumulation |
| [Positional embeddings](docs/layers/positional-embedding.md) | Learned positions, length limits, gradient summation |
| [Single-head attention](docs/layers/attention.md) | Q/K/V projections, masked softmax, backward pass |
| [Multi-head attention](docs/layers/multihead-attention.md) | Head splitting, output projection, backward pass |
| [Layer normalization](docs/layers/normalization.md) | Statistics, affine parameters, gradient derivation |
| [Residual connections](docs/layers/residual.md) | Nested skip connections and gradient routing |
| [Activations](docs/activations.md) | Every activation and derivative, with equations |
| [Losses and gradients](docs/loss.md) | MSE, BCE, CCE, reduction rules, fused softmax + CCE |
| [Optimizers](docs/optimizers.md) | SGD, Momentum, RMSprop, Adam, defaults, state lifecycle |
| [Initialization](docs/initialization.md) | Default, Kaiming, Glorot, custom initializers |
| [API reference](docs/api.md) | Signatures, defaults, input/output shapes, validation |
| [Training and evaluation](docs/training.md) | Manual mini-batch loops, validation, inference, reproducibility |
| [Attention and Transformers](docs/transformer.md) | Sequence modelling, masks, a trainable Transformer example, generation |
| [Serialization](docs/serialization.md) | JSON schema, compatibility, what is omitted |
| [Examples and troubleshooting](docs/examples-and-troubleshooting.md) | Notebooks, tests, common errors, limitations |

## Repository layout

```text
lib/neurality.py                     # Complete library implementation
assets/logo.png                      # Project logo
examples/titanic_survivor/           # Dataset, training/inference notebooks, saved model
examples/simple_transformer/         # BPE tokenizer and small Transformer notebook
tests/                               # unittest suite
docs/                                # Guides, split by topic
docs/layers/                         # One guide per layer type
```

## Examples

Two notebooks demonstrate the library end to end:

- **Titanic survival classification** — preprocessing, dense classifier, training with BCE,
  and inference from the saved JSON model. See
  [`examples/titanic_survivor/`](examples/titanic_survivor/) and the
  [examples guide](docs/examples-and-troubleshooting.md#titanic-classification).
- **Simple Transformer** — a byte-level BPE tokenizer, a causal language model trained on a
  single paragraph, and greedy generation. See
  [`examples/simple_transformer/`](examples/simple_transformer/) and the
  [Transformer guide](docs/transformer.md#a-complete-small-language-model).

Install notebook dependencies and launch Jupyter from the notebook's directory:

```sh
python -m pip install numpy pandas scikit-learn matplotlib jupyterlab
cd examples/titanic_survivor
python -m jupyterlab
```

## Running the tests

From the repository root, with NumPy installed:

```sh
python -m unittest discover -s tests -v
```

| Test module | Coverage |
|---|---|
| `test_attention.py` | Attention gradients, causal behavior, token learning, backend parity, JSON, pure-Python operation |
| `test_gelu.py` | GELU values, finite-difference derivative, dense backward, serialization |
| `test_multihead_attention.py` | Multi-head gradients, padding/causal masks, validation, save/load |
| `test_transformer_components.py` | Position, normalization, and residual gradients; nested residual validation; backend round trips |

The tests compare analytical gradients with finite differences and compare backends with
floating-point tolerance. They verify implementation behavior, not convergence on arbitrary
datasets.

## Design notes and limitations

- **No automatic differentiation.** Every layer implements an explicit backward pass. Custom
  layers require coordinated changes to construction, forward, backward, backend conversion,
  serialization, and validation.
- **No trainer.** There is no `train()`, `batch_size`, gradient accumulation, early stopping,
  or validation loop. The [training guide](docs/training.md) shows how to write one.
- **No tokenizer.** The BPE tokenizer in the Transformer notebook is example code, not part of
  the library. Persist vocabulary and merge order alongside model weights.
- **No GPU, convolution, recurrence, dropout, or KV cache.** Attention recomputes the full
  context on every forward call.
- **Padding masks affect attention keys and values only.** They do not remove padded positions
  from built-in losses. Padding-aware objectives require a masked loss and gradient in
  application code.
- **JSON is a parameter checkpoint.** Optimizer state, preprocessing, and tokenization are not
  included, so resumed training does not exactly reproduce an uninterrupted run.
- **Pure Python is for clarity and portability.** Some NumPy paths still use Python loops;
  selecting NumPy does not vectorize every operation.
- **Educational scope.** The library demonstrates how forward and backward passes fit together.
  It is not a production training framework.

For common error messages and their fixes, see
[examples and troubleshooting](docs/examples-and-troubleshooting.md#common-problems).