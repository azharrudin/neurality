# Dense layer

[Documentation index](../README.md) · [Layer overview](README.md) · [Activations](../activations.md) ·
[Losses](../loss.md) · [Optimizers](../optimizers.md)

A dense (fully connected) layer maps every input feature to `number` output features. Each
output neuron has its own weight vector and scalar bias, then applies an activation function
elementwise.

```python
model.dense(
    number,
    activation_function,
    derivative=None,
    bias=0,
    initialization=None,
)
```

The method returns the model, so calls can be chained.

## Arguments

| Argument | Type | Rules |
|---|---|---|
| `number` | `int` | Positive; number of output neurons $C$ |
| `activation_function` | callable | Required; applied elementwise |
| `derivative` | callable or `None` | Defaults to `N.Derivations.deriv_<activation name>` |
| `bias` | finite number | Same initial bias for every neuron |
| `initialization` | callable or `None` | `None`, `N.Initializations.kaiming`, `N.Initializations.glorot`, or a custom initializer |

If `derivative` is omitted, the implementation looks up
`getattr(N.Derivations, f"deriv_{activation_function.__name__}")`. Built-in activations resolve
automatically; custom activations need an explicit derivative.

## Shapes

Let $D$ be the input feature width, $C$ the neuron count, and $R$ the flattened row count.

| Input shape | Internal shape | Output shape |
|---|---|---|
| $(D,)$ | $(1,D)$ | $(1,C)$ |
| $(B,D)$ | $(B,D)$ | $(B,C)$ |
| $(B,T,D)$ | $(BT,D)$ | $(B,T,C)$ |

The layer flattens all leading dimensions into rows, computes with a matrix of shape
$(R,D)$, then restores the leading dimensions. Parameters are shared across every row,
including all sequence positions.

To set an input width other than the default of $2$, call `fit(X, y)` before adding the first
dense layer. It sets `INPUT_SIZE` from the last dimension of $X$. If this is not the first
layer, the input width is the previous layer's output width.

## Forward pass

For input rows $X \in \mathbb{R}^{R \times D}$, define one weight vector $w_j \in
\mathbb{R}^{D}$ and bias $b_j \in \mathbb{R}$ for each neuron $j = 1,\dots,C$. Stacking the
weight vectors as columns gives $W = [w_1, \dots, w_C] \in \mathbb{R}^{D \times C}$, and
$b = [b_1,\dots,b_C]^\top \in \mathbb{R}^{C}$. The layer computes

$$
Z = XW + \mathbf{1}_R b^\top,
\qquad
A = f(Z),
$$

where $\mathbf{1}_R$ is a column of $R$ ones, $Z \in \mathbb{R}^{R \times C}$ is the
preactivation matrix, and $f$ is applied elementwise. With softmax, $f$ is applied per row.

The implementation stores one weight vector per neuron, then stacks the vectors into $W$ for
the matrix multiplication. This layout is what `model.perceptrons` exposes; it does not change
the mathematics.

### Softmax output layer

If the activation is the built-in softmax, the forward pass computes probabilities per row:

$$
P_{rj} = \frac{\exp(Z_{rj})}{\sum_{k=1}^{C}\exp(Z_{rk})}.
$$

For numerical stability the implementation subtracts the row maximum before exponentiation;
the result is mathematically identical.

## Backward pass

Let $dA = \partial L / \partial A$ be the incoming gradient. The layer first applies the
activation derivative:

$$
dZ = dA \odot f'(Z),
$$

where $f'$ is evaluated on the preactivation matrix $Z$. For each neuron $j$, the
implementation evaluates the stored derivative on column $Z_{:,j}$.

Parameter and input gradients follow:

$$
dW = X^\top dZ,
\qquad
db = dZ^\top \mathbf{1}_R = \sum_{r=1}^{R} dZ_{r,:},
\qquad
dX = dZ\,W^\top.
$$

Because $b_j$ is a scalar shared by every row, its gradient is the sum of the $j$-th column of
$dZ$. The layer appends $dW[:,j]$ and $db_j$ for each neuron, then passes $dX$ to the previous
layer.

### Softmax handling

A softmax activation couples all classes in a row, so the dense layer does not multiply by a
diagonal derivative. Backward uses the Jacobian–vector product. Let $P = \mathrm{softmax}(Z)$
and let $g$ be one row of the incoming gradient $dA$. Then

$$
dz = P \odot \left(g - \langle P, g\rangle \, \mathbf{1}\right),
\qquad
\langle P,g\rangle = \sum_{k=1}^{C} P_k g_k.
$$

The same rule applies row by row. It is used when the layer's activation is
`N.Activations.softmax` and its derivative is `N.Derivations.deriv_softmax`, but the following
fused shortcut does not apply.

#### Fused softmax and CCE

When all three conditions hold:

1. the dense layer is the **last** layer,
2. it uses the built-in softmax activation with `deriv_softmax`, and
3. `backward()` receives exactly `N.Gradients.gradient_cce` (not a wrapper),

the implementation skips the separate softmax Jacobian and applies

$$
dZ_{rj} = \frac{P_{rj}\sum_{k=1}^{C} Y_{rk} - Y_{rj}}{R},
$$

where $Y$ is the target matrix. For one-hot or otherwise normalized targets,
$\sum_k Y_{rk} = 1$, so this simplifies to

$$
dZ = \frac{P - Y}{R}.
$$

This is the usual numerically stable combination of softmax and cross-entropy. Passing a
wrapper around `gradient_cce`, or placing softmax before the last layer, uses the general
softmax path instead. See [losses and gradients](../loss.md#fused-softmax-and-cce).

## Initialization

| `initialization` | Distribution of each weight |
|---|---|
| `None` | $\mathcal{N}(0, 0.1^2)$ |
| `N.Initializations.kaiming` | $\mathcal{N}\!\left(0, \frac{2}{D}\right)$ |
| `N.Initializations.glorot` | $\mathcal{N}\!\left(0, \frac{2}{D + C}\right)$ |
| custom callable | Return value, validated to have shape $(D,)$ |

The built-in distributions are normal distributions; the second argument is a standard
deviation (equivalently, the variance shown above). Every neuron is initialized independently.

A custom initializer is called as `initialization(fan_in, fan_out)` once per neuron and must
return a finite vector of length `fan_in`. It does not receive a random generator. Because it is
called once per neuron, returning the same vector every time produces identical neurons.

All neurons receive the same initial `bias` value, which must be finite. Biases are trainable
like weights.

## Example

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

print([int(row[0] >= 0.5) for row in model.forward(X)])
# [0, 1, 1, 1]
```

The first dense layer receives width $D=2$ from `fit()` and produces $C=4$ features. The second
receives $D=4$ and produces $C=1$. The sigmoid output pairs with BCE.

## Behavior and constraints

- The layer does not add dropout, weight decay, or batch normalization.
- All neurons in one layer share the activation callable; the loader enforces this when
  reading JSON.
- A custom activation or derivative cannot be saved. Only the seven built-in activations are
  serializable.
- Calling `dense()` before `fit()` on a dense model uses the default `INPUT_SIZE = 2`. Calling
  `fit()` later with a different feature count does not rebuild existing weights, so the next
  forward will fail its shape check.
- The reported loss must be computed separately from `backward()`; see
  [training](../training.md) and [losses](../loss.md).

## Related guides

- [Activations and derivatives](../activations.md)
- [Losses and gradients](../loss.md)
- [Optimizers](../optimizers.md)
- [Initialization](../initialization.md)
- [API reference](../api.md#layer-construction)