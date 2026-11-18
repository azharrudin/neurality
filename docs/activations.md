# Activations and derivatives

[Documentation index](README.md) · [Dense layer](layers/dense.md) · [Losses](loss.md)

Dense layers apply an activation function elementwise. Each dense neuron stores both the
activation $f$ and its derivative $f'$. The derivative is evaluated on the **preactivation**
$Z$, not on the activated output $A$.

```python
model.dense(16, N.Activations.gelu)                    # derivative resolved automatically
model.dense(16, N.Activations.gelu, N.Derivations.deriv_gelu)
```

When `derivative` is omitted, the builder resolves it by name:

```python
derivative = getattr(N.Derivations, f"deriv_{activation_function.__name__}")
```

## Built-in activations

| Constant | $f(z)$ | $f'(z)$ |
|---|---|---|
| `N.Activations.linear` | $z$ | $1$ |
| `N.Activations.relu` | $\max(0,z)$ | $\mathbf{1}[z > 0]$ |
| `N.Activations.leaky_relu` | $z$ if $z > 0$, else $0.01z$ | $1$ if $z > 0$, else $0.01$ |
| `N.Activations.sigmoid` | $\dfrac{1}{1+e^{-z}}$ | $\sigma(z)\,(1-\sigma(z))$ |
| `N.Activations.tanh` | $\tanh(z)$ | $1-\tanh^2(z)$ |
| `N.Activations.gelu` | $\tfrac{1}{2}z\left(1+\tanh\!\left(\sqrt{\tfrac{2}{\pi}}\left(z+0.044715z^3\right)\right)\right)$ | see [GELU](#gelu) |
| `N.Activations.softmax` | $\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ per row | see [Softmax](#softmax) |

The matching derivative constants are `N.Derivations.deriv_linear`, `deriv_relu`,
`deriv_leaky_relu`, `deriv_sigmoid`, `deriv_tanh`, `deriv_gelu`, and `deriv_softmax`.

### Piecewise and threshold conventions

- ReLU returns exactly zero at $z=0$; its derivative also returns $0$ at $z=0$ because the
  condition is strict ($z>0$).
- Leaky ReLU uses slope $0.01$ for non-positive inputs and derivative $0.01$ for $z\le 0$.
- `linear` returns the input unchanged in value, converted to the selected backend
  representation.

### Sigmoid stability

The implementation evaluates the sigmoid in a sign-dependent form:

$$
\sigma(z) =
\begin{cases}
\dfrac{1}{1+e^{-z}}, & z \ge 0, \\[6pt]
\dfrac{e^{z}}{1+e^{z}}, & z < 0.
\end{cases}
$$

Both branches are algebraically the same function, but the second avoids computing
$e^{-z}$ for a large positive exponent. `deriv_sigmoid` first computes the stable activation,
then returns $a(1-a)$.

### Tanh

$$
\tanh(z) = \frac{e^{z}-e^{-z}}{e^{z}+e^{-z}},
\qquad
\frac{d}{dz}\tanh(z) = 1-\tanh^2(z).
$$

Both the pure-Python and NumPy paths use the standard-library/NumPy hyperbolic tangent.

## GELU

Neurality uses the tanh approximation of the Gaussian Error Linear Unit, the variant used by
GPT-1:

$$
\operatorname{GELU}(z)
= \frac{1}{2}z\left(1+\tanh\!\left(\sqrt{\frac{2}{\pi}}\left(z+0.044715z^3\right)\right)\right).
$$

Let

$$
t = \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(z+0.044715z^3\right)\right).
$$

Then the derivative implemented by `deriv_gelu` is

$$
\frac{d}{dz}\operatorname{GELU}(z)
= \frac{1}{2}(1+t)
+ \frac{1}{2}z\,(1-t^2)\sqrt{\frac{2}{\pi}}\left(1+3\cdot 0.044715\,z^2\right).
$$

This is the exact derivative of the tanh approximation above, including the cubic term's
contribution. It is not the derivative of the exact Gaussian CDF form of GELU.

## Softmax

Softmax converts a row of logits to a probability distribution:

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}.
$$

The implementation subtracts the row maximum before exponentiation:

$$
p_i = \frac{e^{z_i - m}}{\sum_{j=1}^{C} e^{z_j - m}},
\qquad
m = \max_j z_j.
$$

This is mathematically identical and avoids overflow.

### Jacobian

Softmax couples every output to every input in the same row. Its Jacobian is

$$
J_{ij} = \frac{\partial p_i}{\partial z_j} = p_i\left(\delta_{ij} - p_j\right),
$$

where $\delta_{ij}$ is the Kronecker delta. `N.Derivations.deriv_softmax(x)` returns one
$C \times C$ Jacobian per row, producing shape $(R,C,C)$ for a 2D input.

### Backward used by dense layers

Dense layers do not multiply by the diagonal of the Jacobian. For an incoming row gradient $g$
with respect to the probabilities, the Jacobian–vector product is

$$
dz_i = \sum_{j=1}^{C} J_{ij} g_j
= p_i\left(g_i - \sum_{j=1}^{C} p_j g_j\right),
$$

or, in vector form,

$$
dz = p \odot \left(g - \langle p, g\rangle\,\mathbf{1}\right).
$$

This is the general softmax backward path. When the softmax layer is last and the gradient
function is exactly `N.Gradients.gradient_cce`, the layer instead uses a fused shortcut that
bypasses the probability-space gradient entirely. See
[dense: fused softmax and CCE](layers/dense.md#fused-softmax-and-cce) and
[losses](loss.md#fused-softmax-and-cce).

### Direct calls

`N.Activations.softmax(x)` requires a 2D matrix. Dense layers flatten their input to rows before
calling it, so sequence models can still use softmax over the final feature axis. To apply
softmax to a single vector, wrap it:

```python
N.Activations.softmax([[0.2, 0.4, -0.1]])
```

## Custom activations

A custom activation must be a callable and must be paired with a derivative:

```python
def square_plus(z):
    return z * z + z

def d_square_plus(z):
    return 2 * z + 1

model.dense(8, square_plus, d_square_plus)
```

Rules:

- The derivative receives preactivation values, exactly like the built-ins.
- Both functions must work on the backend's representation (nested lists or NumPy arrays).
- Automatic name lookup only works when a `N.Derivations.deriv_<name>` attribute exists.
- A custom activation or derivative cannot be saved to JSON. Saving raises
  `ValueError("save only supports built-in Neurality activations and derivatives")`.

## Serialization names

JSON stores the activation and derivative by name, not by code. The loader resolves

| JSON field | Looked up as |
|---|---|
| `"activation": "gelu"` | `N.Activations.gelu` |
| `"derivative": "deriv_gelu"` | `N.Derivations.deriv_gelu` |

Only the seven built-in pairs can be reconstructed.

## Related guides

- [Dense layer](layers/dense.md)
- [Losses and gradients](loss.md)
- [API reference](api.md#layer-construction)
- [Serialization](serialization.md)