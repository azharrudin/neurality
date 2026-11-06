# Neurality

![Neurality logo](assets/logo.png)

Neurality is a neural network implementation with dense layers, token embeddings, and single-head self-attention for learning forward propagation, backpropagation, and weight optimization. Models support training and inference with either NumPy or pure Python.

## Features

- Dense layers with configurable neuron counts through `dense()`.
- Trainable token embeddings and single-head self-attention with optional causal masking.
- ReLU, Leaky ReLU, sigmoid, tanh, linear, and softmax activation.
- MSE, binary cross-entropy (BCE), and categorical cross-entropy (CCE) losses.
- SGD, Momentum, RMSprop, and Adam optimizers.
- Kaiming/He normal and Glorot/Xavier normal weight initialization.
- A pure Python backend without NumPy and a NumPy backend with array-based matrix operations.
- JSON model serialization compatible with both backends.

## Project structure

```text
lib/
    neurality.py
examples/
    titanic_survivor/
        training.ipynb
        inference.ipynb
        Titanic-Dataset.csv
        model.json
assets/
```

The main implementation is in [lib/neurality.py](lib/neurality.py). See [training.ipynb](examples/titanic_survivor/training.ipynb) for a Titanic classification experiment.

## Setup

Run the README examples from the project root. Neurality does not require a package installation; import it directly from the `lib` directory:

```python
from lib import neurality
```

The Python backend requires only the Python standard library. For the NumPy backend, which is the default, install NumPy:

```sh
python -m pip install numpy
```

To open and run the Titanic training notebook:

```sh
python -m pip install numpy pandas scikit-learn matplotlib jupyterlab
cd examples/titanic_survivor
python -m jupyterlab
```

The training notebook uses the relative path `../../lib` and reads the CSV from the example directory. When using the Python backend, convert predictions with `np.asarray(...)` before calling NumPy methods such as `.ravel()` in the visualization or accuracy cells.

## Binary classification quickstart

This example learns the logical OR function using pure Python without NumPy:

```python
import random
from lib import neurality

random.seed(42)

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [1.0]]

model = neurality.Neurality(backend=neurality.Backends.PYTHON).fit(X, y)
model.dense(
    4,
    model.Activations.tanh,
    initialization=model.Initializations.glorot,
)
model.dense(
    1,
    model.Activations.sigmoid,
    initialization=model.Initializations.glorot,
)
model.optimizer(neurality.Optimizers.Adam(learning_rate=0.03))

for epoch in range(500):
    prediction = model.forward(X)
    loss = model.Loss.loss_bce(prediction, y)
    model.backward(y, model.Gradients.gradient_bce)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: loss={loss:.6f}")

probabilities = model.forward(X)
labels = [int(row[0] >= 0.5) for row in probabilities]
accuracy = sum(label == target[0] for label, target in zip(labels, y)) / len(y)
print(labels)
print(f"Accuracy: {accuracy:.2%}")
```

`fit()` stores the data and sets the input feature count; it does not train the model. Call `fit()` before adding layers so their weight dimensions match the data. Without `fit()`, the default input size is two.

## Selecting a backend

Set the default backend for models created afterward:

```python
neurality.backend(neurality.Backends.PYTHON)
model = neurality.Neurality()
```

To use NumPy:

```python
neurality.backend(neurality.backends.NUMPY)
model = neurality.Neurality()
```

`Backends` and `backends` refer to the same enum. Changing the global default does not affect existing models. Select a backend for an individual model through its constructor or `backend()` method:

```python
model = neurality.Neurality(backend=neurality.Backends.PYTHON)
model.backend(neurality.Backends.NUMPY)
```

| Backend | Data/output representation | Dependencies |
|---|---|---|
| `PYTHON` | Python lists | Standard library |
| `NUMPY` | NumPy arrays | NumPy |

If NumPy is not installed, select the Python backend in the constructor or set the global default before creating a model. Calling `Neurality()` with the NumPy default requires NumPy before `model.backend(...)` can run.

Switching backends converts weights and data, resets optimizer state, and clears the forward cache. Run `forward()` again before calling `backward()`.

## Layers, activations, and initialization

Add a layer with:

```python
model.dense(
    number=16,
    activation_function=model.Activations.relu,
    derivative=model.Derivations.deriv_relu,
    bias=0.0,
    initialization=model.Initializations.kaiming,
)
```

For built-in activations, omit `derivative` to let Neurality find it in `Derivations`. Each neuron receives the initial bias specified by `bias`.

| Activation | Derivative |
|---|---|
| `Activations.relu` | `Derivations.deriv_relu` |
| `Activations.leaky_relu` | `Derivations.deriv_leaky_relu` |
| `Activations.sigmoid` | `Derivations.deriv_sigmoid` |
| `Activations.tanh` | `Derivations.deriv_tanh` |
| `Activations.linear` | `Derivations.deriv_linear` |
| `Activations.softmax` | `Derivations.deriv_softmax` |

Leaky ReLU uses a slope of `0.01` for negative values. Softmax operates on each sample row. Backpropagation accounts for dependencies between softmax outputs and supports a fused softmax + CCE path.

| Initialization | Standard deviation | Typical use |
|---|---|---|
| `Initializations.kaiming` | `sqrt(2 / fan_in)` | ReLU |
| `Initializations.glorot` | `sqrt(2 / (fan_in + fan_out))` | Tanh or linear |
| `None` | `0.1` | Default |

`fan_in` is the number of layer inputs, and `fan_out` is the number of neurons in the layer. A custom `initializer(fan_in, fan_out)` must return a finite weight vector of length `fan_in` for one neuron.

## Losses and gradients

| Task | Output | Loss | Gradient |
|---|---|---|---|
| Binary classification | Sigmoid | `Loss.loss_bce` | `Gradients.gradient_bce` |
| Multiclass classification | Softmax | `Loss.loss_cce` | `Gradients.gradient_cce` |
| Regression or reconstruction | Linear, or an activation matching the target range | `Loss.loss_mse` | `Gradients.gradient_mse` |

Dense inputs use shape `(samples, features)` or `(batch, sequence, features)`. Dense layers share their weights across sequence positions. When computing a loss, use targets with shape `(samples, outputs)`, such as `[[0], [1]]` for a single output. CCE expects a target distribution over classes, typically one-hot vectors such as `[[1, 0, 0], [0, 1, 0]]`.

`forward()` caches activations for backpropagation. Call `backward()` after forwarding the same training batch; evaluating other data between these calls replaces the cache. Built-in gradients are already normalized, so do not divide them by the batch size again.

## Embeddings and self-attention

`embedding(vocab_size, embedding_size)` must be the first layer. It accepts integer token IDs with shape `(batch, sequence)` and returns vectors with shape `(batch, sequence, embedding_size)`. A one-dimensional input is treated as one sequence. Token IDs must be in `[0, vocab_size)`.

`attention(key_size=None, causal=False, embedding_size=None)` adds single-head scaled dot-product self-attention. It learns query, key, and value matrices. The output feature size matches the input feature size. `key_size` defaults to that size; `causal=True` prevents each token from attending to later positions.

```python
from lib import neurality

model = neurality.Neurality(backend=neurality.Backends.PYTHON)
model.embedding(vocab_size=4, embedding_size=8)
model.attention(key_size=8, causal=True)
model.dense(4, model.Activations.softmax)
model.optimizer(neurality.Optimizers.Adam(learning_rate=0.01))

tokens = [[0, 1, 2], [1, 2, 0]]
next_tokens = [[1, 2, 0], [2, 0, 1]]
targets = [
    [[float(token == candidate) for candidate in range(4)] for token in sequence]
    for sequence in next_tokens
]

for epoch in range(100):
    prediction = model.forward(tokens)
    loss = model.Loss.loss_cce(prediction, targets)
    model.backward(targets, model.Gradients.gradient_cce)

prediction = model.forward([[0, 1, 2]])
last_position = prediction[0][-1]
next_id = max(range(len(last_position)), key=lambda i: last_position[i])
print(next_id)
```

The pipeline is `(batch, sequence)` → embedding → attention → dense → `(batch, sequence, vocab_size)`. CCE averages over all batch and sequence positions. Embedding gradients accumulate when token IDs repeat. Optimizers update the embedding table, attention projections, and dense parameters in a single backward step.

Attention can also process existing feature vectors without an embedding layer:

```python
model = neurality.Neurality(backend=neurality.Backends.PYTHON)
model.attention(embedding_size=3, key_size=2, causal=False)
prediction = model.forward([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
```

For standalone attention, a 2D matrix represents one sequence `(sequence, features)`; a 3D tensor represents `(batch, sequence, features)`. Within a sequential model, the feature dimension is inferred from the previous layer. `model.layers` stores the execution order; `model.perceptrons` continues to contain only dense neuron lists.

This is a basic self-attention layer, not a complete Transformer block. There is no positional encoding, multi-head attention, padding mask, residual connection, or LayerNorm. Tokenization, positional information, and text generation loops must be handled separately.

## Optimizers

Choose one optimizer before the training loop:

```python
model.optimizer(neurality.Optimizers.SGD(learning_rate=0.01))
model.optimizer(neurality.Optimizers.Momentum(learning_rate=0.01, momentum=0.9))
model.optimizer(neurality.Optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-8))
model.optimizer(neurality.Optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8))
```

These four lines show alternatives. Each call to `optimizer()` replaces the previous optimizer and resets its state. Do not set it again at every epoch. The same classes are also available through `model.Optimizers` and `Neurality.Optimizers`.

If no optimizer is selected, the model uses SGD with a learning rate of `0.1`. The `learning_rate` argument to `backward()` overrides the rate for that update only:

```python
model.backward(y, model.Gradients.gradient_bce, learning_rate=0.01)
```

## Mini-batch

There is no built-in `train()` method or `batch_size` parameter. Split the data in your training loop. This example continues with the model and the `X` and `y` data from the quickstart and works with both backends:

```python
batch_size = 2
rng = random.Random(42)

for epoch in range(100):
    indices = list(range(len(X)))
    rng.shuffle(indices)
    total_loss = 0.0

    for start in range(0, len(X), batch_size):
        batch = indices[start:start + batch_size]
        X_batch = [X[i] for i in batch]
        y_batch = [y[i] for i in batch]

        prediction = model.forward(X_batch)
        total_loss += model.Loss.loss_bce(prediction, y_batch) * len(batch)
        model.backward(y_batch, model.Gradients.gradient_bce)

    print(f"Epoch {epoch + 1}: loss={total_loss / len(X):.6f}")
```

Use `batch_size=1` for one sample per update or `batch_size=len(X)` for full-batch training. The final batch may contain fewer samples.

## Inference

Pass a single sample as a vector or a matrix containing one row:

```python
prediction = model.forward([0.0, 1.0])
probability = float(prediction[0][0])
label = int(probability >= 0.5)
print(probability, label)
```

Results have shape `(samples, outputs)` with both backends. For multiclass output, select the index with the highest probability:

```python
label = max(range(len(prediction[0])), key=lambda i: prediction[0][i])
```

Use the same feature count, column order, encoding, and normalization as during training. Compute normalization statistics from the training data and reuse them for validation and inference.

## Saving and loading JSON models

```python
model.save("model.json")

loaded = neurality.Neurality.load(
    "model.json",
    backend=neurality.Backends.PYTHON,
)
print(loaded.forward([0.0, 1.0]))
```

`load()` returns a new model; assign its result to a variable. Models saved with NumPy can be loaded with Python and vice versa.

The JSON file stores the input feature count, layer types and configuration, weights, biases, activation names, and derivative names, including embedding tables and attention projections. New saves use format version 2; version 1 dense models remain loadable. Only built-in activations and derivatives are supported. Initializers are not needed when loading because the trained weights are already stored.

Optimizer state and configuration, training data, preprocessing, class labels, and tokenizers are not saved. Loaded models use the default optimizer; select an optimizer again to continue training with fresh state. Store preprocessing configuration separately.

## Current limitations

Neurality supports dense networks, token embeddings, and single-head self-attention. It does not currently provide convolution, recurrent layers, complete Transformer blocks, a GPU backend, or general-purpose automatic differentiation. Dense autoencoders can be assembled from the available layers.

Training uses manual loops, including batching, validation, and early stopping. Keep training and test data separate to evaluate performance on unseen samples. The Python backend allows models to run without NumPy, but does not guarantee compatibility with MicroPython or every microcontroller.
