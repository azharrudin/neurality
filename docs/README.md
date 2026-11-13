# Neurality documentation

These guides describe the implementation in [lib/neurality.py](../lib/neurality.py). The [project README](../README.md) contains installation instructions and a runnable first example.

## Reading path

1. [Architecture and mathematics](architecture.md): follow tensors, parameters, and gradients through the implementation.
2. [API reference](api.md): check method signatures, defaults, input shapes, and constraints.
3. [Training and evaluation](training.md): build manual training loops and evaluate predictions.
4. [Attention and Transformers](transformer.md): construct sequence models, apply masks, and generate tokens.
5. [Serialization](serialization.md): understand what model files preserve and how to reload them.
6. [Examples and troubleshooting](examples-and-troubleshooting.md): run the notebooks and diagnose errors.

## Conventions

Code uses `from lib.neurality import Neurality as N`. Here, `N` is the class and `model` is an instance. Run standalone examples from the repository root unless stated otherwise. Fragments explicitly marked as continuations reuse variables from preceding examples.

| Symbol | Meaning |
|---|---|
| `B` | Batch size |
| `T` | Sequence length |
| `D` | Feature count / embedding width |
| `C` | Output count / number of classes |
| `V` | Vocabulary size |
| `H` | Number of attention heads |
| `R` | Number of rows after flattening batch and sequence dimensions |

A batch must be rectangular: all sequences in one tensor have equal lengths. Your application prepares padding, target encodings, preprocessing, and metrics. Comparisons between backends should allow floating-point tolerance.

Methods or attributes prefixed with `_` are implementation details. Descriptions of internal state explain how the library works; application code should prefer the documented construction, forward/backward, and persistence methods.
