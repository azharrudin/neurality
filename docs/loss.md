# Losses and gradients

[Documentation index](README.md) · [Dense layer](layers/dense.md) · [Activations](activations.md) ·
[Optimizers](optimizers.md) · [Training](training.md)

Neurality separates **reporting losses** (`N.Loss`) from **output gradients**
(`N.Gradients`). `forward()` does not compute a loss; you call a loss function yourself, and you
pass the matching gradient function to `backward()`.

```python
prediction = model.forward(batch)
loss = N.Loss.loss_bce(prediction, target)          # reporting only
model.backward(target, N.Gradients.gradient_bce)    # gradients + optimizer update
```

The gradient functions return $\partial L / \partial p$, where $p$ is the model prediction, not
the logits. Built-in gradients already include the reduction factor.

## Pairing table

| Loss | Gradient | Target encoding | Typical output layer |
|---|---|---|---|
| `N.Loss.loss_mse(p, y)` | `N.Gradients.gradient_mse` | Continuous values of the output shape | Linear or range-matched activation |
| `N.Loss.loss_bce(p, y)` | `N.Gradients.gradient_bce` | Values in $[0,1]$ | Sigmoid |
| `N.Loss.loss_cce(p, y)` | `N.Gradients.gradient_cce` | One-hot vectors or probability distributions | Softmax |

## Reductions

Let $E$ be the total number of scalar elements in the prediction and $R$ the number of rows
after flattening all leading dimensions. For a prediction of shape $(B,T,C)$, for example,
$E = BTC$ and $R = BT$.

| Loss | Sum over | Divided by |
|---|---|---|
| MSE | all $E$ elements | $E$ |
| BCE | all $E$ elements | $E$ |
| CCE | all classes, then all $R$ rows | $R$ |

A CCE computation on shape $(B,T,V)$ therefore returns a scalar that is the mean
cross-entropy per token position, not per sequence.

## Mean squared error

$$
L_{\mathrm{MSE}}
= \frac{1}{E}\sum_{i=1}^{E}\left(p_i - y_i\right)^2
$$

The matching gradient is

$$
\frac{\partial L_{\mathrm{MSE}}}{\partial p_i}
= \frac{2}{E}\left(p_i - y_i\right).
$$

There is no clipping or output activation assumption. The target must have the same shape as
the prediction to avoid broadcasting.

## Binary cross-entropy

$$
L_{\mathrm{BCE}}
= -\frac{1}{E}\sum_{i=1}^{E}
\left[
y_i \log p_i + (1-y_i)\log(1-p_i)
\right]
$$

Before evaluating the logarithm, probabilities are clipped to

$$
p_i \leftarrow \mathrm{clip}\left(p_i,\ 10^{-8},\ 1-10^{-8}\right).
$$

The gradient uses the same clipped values:

$$
\frac{\partial L_{\mathrm{BCE}}}{\partial p_i}
= \frac{1}{E}\left(
-\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i}
\right).
$$

BCE supports both a single binary output of shape $(B,1)$ and multiple independent binary
labels of shape $(B,C)$. It does not enforce that targets lie in $[0,1]$, but values outside
that interval are not meaningful for a probabilistic output.

## Categorical cross-entropy

For prediction rows $p_r \in \mathbb{R}^{C}$ and target rows $y_r \in \mathbb{R}^{C}$:

$$
L_{\mathrm{CCE}}
= -\frac{1}{R}\sum_{r=1}^{R}\sum_{j=1}^{C}
y_{rj}\log p_{rj}.
$$

Probabilities are clipped to $[10^{-8}, 1]$ inside the logarithm. The gradient with respect to
the probabilities is

$$
\frac{\partial L_{\mathrm{CCE}}}{\partial p_{rj}}
= -\frac{1}{R}\cdot\frac{y_{rj}}{p_{rj}},
$$

using the clipped $p_{rj}$.

Notes:

- **Targets are distributions, not class IDs.** A target row $[0,1,0]$ means class 1. The
  library does not convert integer labels to one-hot vectors.
- **The target is not validated as a probability distribution.** It is used as-is in the
  weighted sum. Normalized one-hot targets are the usual choice.
- **The sum is over classes, then averaged over rows.** This is the standard token-level
  cross-entropy for language models.

## Fused softmax and CCE

The dense layer has a special backward path when **all** of the following hold:

1. The dense layer is the last layer in `model.layers`.
2. Its activation is exactly `N.Activations.softmax`.
3. Its derivative is exactly `N.Derivations.deriv_softmax`.
4. `backward()` receives exactly `N.Gradients.gradient_cce`.

The layer then skips the probability-space gradient and directly computes the logit gradient

$$
dZ_{rj}
= \frac{P_{rj}\sum_{k=1}^{C}Y_{rk} - Y_{rj}}{R},
$$

where $P = \mathrm{softmax}(Z)$. For a normalized target row,
$\sum_k Y_{rk} = 1$, so

$$
dZ = \frac{P - Y}{R}.
$$

This is the standard numerically stable combination. It also avoids the round trip through
$-\frac{1}{R}\frac{Y}{P}$ and the softmax Jacobian.

Passing a **wrapper** around `gradient_cce`, or using softmax somewhere other than the last
layer, selects the general softmax path. In that case the callback receives and returns
gradients with respect to **probabilities**, and the layer applies the Jacobian–vector product:

$$
dZ = P \odot \left(dA - \langle P, dA\rangle\,\mathbf{1}\right).
$$

See [activations: softmax](activations.md#backward-used-by-dense-layers) and
[dense layer](layers/dense.md#fused-softmax-and-cce).

## Target shapes

Use a target with the exact shape of the prediction.

```python
# One binary output per sample
y = [[0.0], [1.0], [1.0], [1.0]]

# Three mutually exclusive classes
y = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

# Next-token targets for shape (B, T, V)
targets = [
    [[0.0, 1.0, 0.0, 0.0] for _ in range(T)]
    for _ in range(B)
]
```

`backward()` accepts a target with the same number of elements as the output and reshapes it to
the output shape. Loss functions in `N.Loss` do **not** reshape; they rely on the shapes you
pass. In NumPy, a flat target can broadcast against a 2D prediction and silently change the
reduction. Use nested targets such as `[[0.0], [1.0]]` rather than `[0.0, 1.0]` for a single
binary output.

## Masking and padding

Built-in losses ignore `padding_mask` entirely. They average over every position in the tensor,
including padded positions and padding targets. Consequences:

- Setting padded CCE targets to all-zero vectors removes their numerator contribution but not
  their place in the denominator $R$.
- The desired target mask for next-token prediction can differ from the input mask after
  shifting targets by one position.
- A padding-aware objective needs a custom loss and a custom gradient callback that uses the
  same valid-position count.

A custom callback receives flattened prediction and target rows of shape $(R,C)$ and must return
the same shape. For masked CCE, a callback can return

$$
dA_{rj} =
\begin{cases}
-\dfrac{y_{rj}}{p_{rj}\,n_{\text{valid}}}, & \text{row } r \text{ is valid}, \\[6pt]
0, & \text{otherwise},
\end{cases}
$$

where $n_{\text{valid}}$ is the number of included target positions. Report the loss with the
same denominator. Because this callback is not the exact `gradient_cce` function, the general
softmax path applies: return gradients with respect to probabilities, not the fused logit
gradient.

See [Transformer loss masking](transformer.md#loss-masking) for a worked discussion.

## Custom gradient functions

A custom callback `gradient_function(prediction_rows, target_rows)` must:

- accept arrays or lists of shape $(R,C)$, where leading batch/sequence dimensions are
  flattened;
- return a finite value of the same shape;
- represent $\partial L / \partial p$, not $\partial L / \partial Z$;
- include the intended reduction factor.

It is called once per `backward()`, before any parameter update. The returned gradient is
reshaped back to the model output shape and propagated through the layers.

## Reporting versus training

The reported loss and the parameter update use the same prediction only if you call the loss
before `backward()`:

```python
prediction = model.forward(batch)
loss = N.Loss.loss_cce(prediction, target)
model.backward(target, N.Gradients.gradient_cce)   # updates parameters
```

After `backward()`, `prediction` is stale with respect to the updated parameters. For an
evaluation metric at one fixed parameter state, run a separate forward pass after training.

## Related guides

- [Dense layer](layers/dense.md)
- [Activations](activations.md)
- [Optimizers](optimizers.md)
- [Training and evaluation](training.md)
- [Transformer loss masking](transformer.md#loss-masking)
- [API reference](api.md#losses)