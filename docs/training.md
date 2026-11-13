# Training and evaluation

[Documentation index](README.md)

## Choose an output and objective

| Task | Final activation | Loss / gradient | Target shape |
|---|---|---|---|
| Binary classification | Sigmoid | BCE / `gradient_bce` | `(B,1)` |
| Independent binary labels | Sigmoid | BCE / `gradient_bce` | `(B,C)` |
| Mutually exclusive classes | Softmax | CCE / `gradient_cce` | `(B,C)` one-hot or distributions |
| Regression | Linear, or an activation matching the target range | MSE / `gradient_mse` | `(B,C)` |
| Next-token prediction | Softmax | CCE / `gradient_cce` | `(B,T,V)` |

Use corresponding functions under `N.Loss` and `N.Gradients`. CCE does not convert integer class IDs into one-hot targets. For a single binary output, use `[[0.0],[1.0]]` rather than a flat target list; this also avoids accidental NumPy broadcasting in standalone loss calls.

## Prepare the data

Split training, validation, and test sets before fitting preprocessing. Learn means, standard deviations, category encodings, and vocabulary from the training set; reuse the same mappings on validation and inference data. Preserve feature order. Handle missing values and zero standard deviations explicitly.

`fit(X,y)` stores arrays and establishes the input feature count. It does not shuffle, normalize, split, or optimize data. For a dense network, call it before adding layers. Building token models directly with `embedding()` avoids treating sequence length as a dense feature count.

## A complete mini-batch loop

This small OR example demonstrates batching and sample-weighted epoch loss. It uses the same four points for demonstration only; it is not a generalization benchmark.

```python
import random
from lib.neurality import Neurality as N

random.seed(42)
rng = random.Random(42)
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [1.0]]
model = N(backend=N.Backends.PYTHON).fit(X, y)
model.dense(4, N.Activations.tanh, initialization=N.Initializations.glorot)
model.dense(1, N.Activations.sigmoid, initialization=N.Initializations.glorot)
model.optimizer(N.Optimizers.Adam(learning_rate=0.03))
batch_size = 2

for epoch in range(300):
    order = list(range(len(X)))
    rng.shuffle(order)
    total_loss = 0.0
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        xb = [X[i] for i in indices]
        yb = [y[i] for i in indices]
        prediction = model.forward(xb)
        total_loss += N.Loss.loss_bce(prediction, yb) * len(indices)
        model.backward(yb, N.Gradients.gradient_bce)
    if (epoch + 1) % 100 == 0:
        print(epoch + 1, total_loss / len(X))
```

Every mini-batch triggers one optimizer update. The final batch may be smaller. Set `batch_size=1` for online updates or `batch_size=len(X)` for full-batch updates. There is no built-in `train()`, `batch_size`, or gradient accumulation interface.

The reported epoch loss combines predictions made at different parameter states during the epoch. For a metric at one fixed parameter state, run another forward over the evaluation set after updates. For token losses with varying sequence lengths across batches, weight by the number of included token positions instead of only the number of samples.

## Forward cache discipline

Always keep each training forward next to its backward:

```text
forward(training_batch)
compute reporting loss
backward(training_targets, matching_gradient)
forward(validation_batch)
compute validation metrics
```

Do not run validation between the training forward and backward: forward replaces the caches. Use one fresh forward per parameter update. Built-in gradients are already normalized; do not divide them again.

## Validation and early stopping

Validation is another forward pass and does not update weights. There is no `train()`/`eval()` mode switch; current layers have no dropout or running batch statistics.

The following fragment continues the mini-batch example. Replace `X_val` and `y_val` with your separate validation split in a real experiment:

```python
X_val, y_val = X, y  # Demonstration only; use held-out data for real evaluation.
validation_prediction = model.forward(X_val)
validation_loss = N.Loss.loss_bce(validation_prediction, y_val)
labels = [int(row[0] >= 0.5) for row in validation_prediction]
accuracy = sum(a == b[0] for a, b in zip(labels, y_val)) / len(y_val)
print(validation_loss, accuracy)
```

For early stopping, track the best validation loss after each epoch, call `model.save(best_path)` when it improves, and stop after a chosen number of epochs without improvement. Reload the best model before final test evaluation. Saving only at the end retains the final epoch rather than the best one. Optimizer state is not included in these files.

Accuracy can hide class imbalance. Choose metrics appropriate to the task and compute them in application code. Evaluate the test set after model selection, rather than using test performance to tune hyperparameters.

## Inference

For binary classification, take `float(prediction[0][0])` and apply a selected threshold. For multiclass output, choose the largest probability:

```python
prediction = model.forward([0.0, 1.0])
probability = float(prediction[0][0])
label = int(probability >= 0.5)
print(label)
```

For a multiclass row `row`, use `max(range(len(row)), key=lambda i: row[i])`. For regression, interpret outputs in the target's training scale and invert target preprocessing if needed. Sequence models return a distribution at each token position.

## Reproducibility and backend changes

The Python backend uses the standard-library random generator for built-in initialization; NumPy uses `numpy.random`. Seed the relevant generator before building layers. For reproducible experiments using both, seed both:

```python
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```

Matching seeds across different generators do not guarantee identical initial weights. For a numerical comparison, construct one model and convert/load its parameters on the other backend. Switching with `model.backend(...)` resets optimizer state and clears caches, even when reselecting a backend; avoid doing it inside the training loop.

If loss diverges, check target shape, preprocessing, activation/loss pairing, and learning rate. Smaller rates can help, but no single rate guarantees convergence. Token padding requires special care: the attention mask does not mask the objective. See the [Transformer guide](transformer.md).
