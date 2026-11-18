# Neurality documentation

These guides describe the implementation in [`lib/neurality.py`](../lib/neurality.py). The
[project README](../README.md) contains installation instructions and a runnable first example.

## Documentation map

### Core concepts

| Guide | What it explains |
|---|---|
| [Architecture and execution](architecture.md) | How a model is represented, how tensors flow, how backward updates parameters |
| [Backends](backends.md) | NumPy vs native Python: representation, selection, conversion, parity, performance |
| [API reference](api.md) | Constructor and method signatures, defaults, shapes, validation rules |
| [Initialization](initialization.md) | Weight initialization strategies and layer-specific defaults |

### Layers

| Guide | Layer |
|---|---|
| [Layer overview](layers/README.md) | Common conventions and a comparison table |
| [Dense](layers/dense.md) | Fully connected layer and activation application |
| [Embedding](layers/embedding.md) | Trainable token lookup table |
| [Positional embedding](layers/positional-embedding.md) | Learned position vectors |
| [Attention](layers/attention.md) | Single-head self-attention |
| [Multi-head attention](layers/multihead-attention.md) | Parallel heads plus output projection |
| [Layer normalization](layers/normalization.md) | Per-token feature normalization |
| [Residual connections](layers/residual.md) | Skip connections with nested pairing |

### Mathematical components

| Guide | Contents |
|---|---|
| [Activations](activations.md) | Forward definitions and derivatives for all built-in activations |
| [Losses and gradients](loss.md) | MSE, BCE, CCE, reduction conventions, fused softmax + CCE |
| [Optimizers](optimizers.md) | SGD, Momentum, RMSprop, Adam, update equations, state lifecycle |

### Workflows

| Guide | Contents |
|---|---|
| [Training and evaluation](training.md) | Manual training loops, batching, validation, inference, reproducibility |
| [Attention and Transformers](transformer.md) | Sequence models, masks, a trainable Transformer example, generation |
| [Serialization](serialization.md) | JSON schema, compatibility, and omitted state |
| [Examples and troubleshooting](examples-and-troubleshooting.md) | Notebooks, tests, common errors, limitations |

## Reading paths

- **First contact:** run the [README quickstart](../README.md#quickstart-binary-classification),
  then read [architecture](architecture.md) and the [dense layer](layers/dense.md).
- **Building a classifier:** read [training](training.md) together with
  [losses](loss.md) and [optimizers](optimizers.md).
- **Building a Transformer:** read the [layer overview](layers/README.md), then
  [embedding](layers/embedding.md), [attention](layers/attention.md),
  [multi-head attention](layers/multihead-attention.md),
  [normalization](layers/normalization.md), and [residual connections](layers/residual.md).
  Finish with [Attention and Transformers](transformer.md).
- **Debugging or extending:** read [architecture](architecture.md) and the
  [API reference](api.md), then the relevant layer page.

## Conventions

All code imports the class with an alias:

```python
from lib.neurality import Neurality as N
```

`N` is the class and `model` is an instance. Standalone examples run from the repository root
unless a guide states otherwise. Fragments marked as continuations reuse variables from the
preceding code block.

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $T$ | Sequence length |
| $D$ | Feature width / embedding width |
| $C$ | Output width / number of classes |
| $V$ | Vocabulary size |
| $H$ | Number of attention heads |
| $d_k$ | Key width in single-head attention |
| $d_h$ | Per-head width, $D/H$ |
| $R$ | Number of rows after flattening batch and sequence dimensions |
| $E$ | Total number of output elements |

Tensors are written as shape tuples such as $(B,T,D)$. Matrices are uppercase ($X$, $W$),
scalars and elementwise outputs are lowercase, and $\odot$ denotes an elementwise product. A
superscript $\top$ denotes a transpose. The symbol $\mathbf{1}$ denotes an all-ones vector
whose length makes the surrounding expression well-defined; in row-wise formulas it acts as a
row of ones.

Shape rules that apply everywhere:

- A batch must be rectangular: all sequences in one tensor have equal length.
- Layers preserve the batch and sequence axes unless a guide says otherwise.
- $D$ is the final feature dimension. Dense layers change $D$ to $C$; sequence layers keep it.
- Application code prepares padding, target encodings, preprocessing, tokenization, and metrics.

Names beginning with `_` are implementation details. Descriptions of internal state explain how
the library works; application code should use the documented construction, forward/backward,
and persistence methods.

Comparisons between backends should allow floating-point tolerance. NumPy may broadcast
operands where pure Python requires equal list shapes, so supply exact target shapes.