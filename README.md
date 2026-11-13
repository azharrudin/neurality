# Neurality

![Neurality logo](assets/logo.png)

Neurality is an educational deep learning library implemented in a single Python file. It exposes forward propagation, backpropagation, and parameter updates for dense networks and Transformer components using either pure Python or NumPy.

Read the [in-depth documentation](docs/README.md) for architecture, API reference, training, Transformers, and serialization.

## Features

- Dense layers with ReLU, Leaky ReLU, GELU, sigmoid, tanh, linear, and softmax activations.
- Trainable token and positional embeddings, single-head and multi-head self-attention.
- Causal and padding masks, layer normalization, and nested residual connections.
- MSE, binary cross-entropy (BCE), and categorical cross-entropy (CCE).
- SGD, Momentum, RMSprop, and Adam; normal, Kaiming, and Glorot initialization.
- JSON model persistence across both backends.

Training loops, batching, evaluation, preprocessing, tokenization, and generation are controlled by your code. There is no GPU backend, general-purpose automatic differentiation, convolutional layer, recurrent layer, or packaged Transformer block.

## Setup

Run examples from the repository root. No Neurality package installation is needed:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
```

NumPy is needed for the default backend and the full test suite. The pure Python backend uses only the standard library. Select it explicitly when NumPy is unavailable.

## Quickstart: binary classification

```python
import random
from lib.neurality import Neurality as N

random.seed(42)
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [1.0]]

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
# Expected labels: [0, 1, 1, 1]
```

`fit()` stores data and sets the input feature count; it does not train the model. For dense networks, call it before adding layers. Training requires a fresh `forward()` followed by `backward()` for each update.

Save and reload the model from the quickstart:

```python
model.save("model.json")
loaded = N.load("model.json", backend=N.Backends.PYTHON)
probability = float(loaded.forward([0.0, 1.0])[0][0])
print(probability, int(probability >= 0.5))
```

## In-depth documentation

Start with the [documentation index](docs/README.md).

| Guide | Contents |
|---|---|
| [Architecture and mathematics](docs/architecture.md) | Internal representation, layer equations, gradients, optimizer updates |
| [API reference](docs/api.md) | Signatures, defaults, dimensions, and validation |
| [Training and evaluation](docs/training.md) | Loss selection, batching, validation, reproducibility, inference |
| [Attention and Transformers](docs/transformer.md) | Token sequences, masking, a trainable Transformer example, generation |
| [Serialization](docs/serialization.md) | JSON schema, compatibility, loading, resuming training |
| [Examples and troubleshooting](docs/examples-and-troubleshooting.md) | Notebooks, testing, common errors, limitations |

## Repository layout

```text
lib/neurality.py                           # Complete library implementation
assets/logo.png                           # Project logo
examples/titanic_survivor/                 # Dataset, training/inference notebooks, model
examples/simple_transformer/               # BPE tokenizer and small Transformer notebook
tests/                                    # unittest suite
docs/                                     # Detailed documentation
```

## Notebooks and tests

```sh
python -m pip install numpy pandas scikit-learn matplotlib jupyterlab
cd examples/titanic_survivor
python -m jupyterlab
```

For the Transformer notebook, launch Jupyter from `examples/simple_transformer` instead. The notebooks use `../../lib` relative to their working directory. See the [examples guide](docs/examples-and-troubleshooting.md) for dependencies and evaluation caveats.

Run tests from the repository root:

```sh
python -m unittest discover -s tests -v
```

JSON files contain architecture and parameters, but omit optimizer state, preprocessing, class labels, and tokenizers. Padding masks affect attention keys/values; they do not exclude padded positions from built-in losses.
