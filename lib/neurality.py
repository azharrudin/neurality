
# =============================================================================
# Neurality — A Lightweight Deep Learning Library
# =============================================================================
# A simple, single-file deep learning library 
# featuring implementations of Dense Layers, Embeddings, and Attention Mechanisms.
#
# Designed for simplicity, readability, and educational purposes.
#
# Author  : UD!N!UM
# Version : 0.1.0
# =============================================================================
import json
import math
import random
from enum import Enum


class Neurality:
    class Backends(Enum):
        PYTHON = "python"
        NUMPY = "numpy"

    backends = Backends
    _default_backend = Backends.NUMPY

    class Math:
        @staticmethod
        def numpy():
            try:
                import numpy
            except ImportError as error:
                raise ImportError("NumPy is required for the NUMPY backend; select Backends.PYTHON instead") from error
            return numpy

        @staticmethod
        def is_numpy(value):
            return type(value).__module__.split(".")[0] == "numpy"

        @staticmethod
        def tolist(value):
            if hasattr(value, "to_numpy"):
                value = value.to_numpy()
            if hasattr(value, "tolist"):
                value = value.tolist()
            if isinstance(value, (list, tuple)):
                return [Neurality.Math.tolist(item) for item in value]
            return float(value)

        @staticmethod
        def shape(value):
            if hasattr(value, "shape"):
                return tuple(value.shape)
            if not isinstance(value, (list, tuple)):
                return ()
            if not value:
                return (0,)
            shape = Neurality.Math.shape(value[0])
            if any(Neurality.Math.shape(item) != shape for item in value):
                raise ValueError("Array rows must have consistent dimensions")
            return (len(value),) + shape

        @staticmethod
        def array(value, backend=None):
            backend = Neurality._default_backend if backend is None else backend
            if backend is Neurality.Backends.NUMPY:
                return Neurality.Math.numpy().array(value, dtype=float, copy=True)
            value = Neurality.Math.tolist(value)
            Neurality.Math.shape(value)
            return value

        @staticmethod
        def reshape(value, shape):
            if math.prod(shape) != len(Neurality.Math.flatten(value)):
                raise ValueError("Cannot reshape data to the requested shape")
            if Neurality.Math.is_numpy(value):
                return value.reshape(shape).copy()
            values = iter(Neurality.Math.flatten(value))
            def build(dimensions):
                if not dimensions:
                    return next(values)
                return [build(dimensions[1:]) for _ in range(dimensions[0])]
            return build(tuple(shape))

        @staticmethod
        def rows(value):
            shape = Neurality.Math.shape(value)
            if len(shape) < 2 or not all(shape):
                raise ValueError("Expected a non-empty matrix or batch of matrices")
            return Neurality.Math.reshape(value, (math.prod(shape[:-1]), shape[-1]))

        @staticmethod
        def flatten(value):
            value = Neurality.Math.tolist(value)
            if isinstance(value, list):
                return [x for item in value for x in Neurality.Math.flatten(item)]
            return [value]

        @staticmethod
        def finite(value):
            if Neurality.Math.is_numpy(value):
                return bool(Neurality.Math.numpy().isfinite(value).all())
            return all(math.isfinite(x) for x in Neurality.Math.flatten(value))

        @staticmethod
        def unary(value, function, numpy_function=None):
            if Neurality.Math.is_numpy(value):
                np = Neurality.Math.numpy()
                if numpy_function is not None:
                    return numpy_function(np, value)
                result = [function(x) for x in value.ravel()]
                return np.asarray(result, dtype=float).reshape(value.shape)
            if isinstance(value, (list, tuple)):
                return [Neurality.Math.unary(item, function) for item in value]
            return function(float(value))

        @staticmethod
        def binary(a, b, function):
            if Neurality.Math.is_numpy(a) or Neurality.Math.is_numpy(b):
                return function(Neurality.Math.numpy().asarray(a), Neurality.Math.numpy().asarray(b))
            a_list, b_list = isinstance(a, (list, tuple)), isinstance(b, (list, tuple))
            if a_list and b_list:
                if Neurality.Math.shape(a) != Neurality.Math.shape(b):
                    raise ValueError("Elementwise operands must have the same shape")
                return [Neurality.Math.binary(x, y, function) for x, y in zip(a, b)]
            if a_list:
                return [Neurality.Math.binary(x, b, function) for x in a]
            if b_list:
                return [Neurality.Math.binary(a, y, function) for y in b]
            return function(a, b)

        @staticmethod
        def add(a, b):
            return Neurality.Math.binary(a, b, lambda x, y: x + y)

        @staticmethod
        def sub(a, b):
            return Neurality.Math.binary(a, b, lambda x, y: x - y)

        @staticmethod
        def mul(a, b):
            return Neurality.Math.binary(a, b, lambda x, y: x * y)

        @staticmethod
        def div(a, b):
            return Neurality.Math.binary(a, b, lambda x, y: x / y)

        @staticmethod
        def square(a):
            return Neurality.Math.mul(a, a)

        @staticmethod
        def sqrt(a):
            return Neurality.Math.unary(a, math.sqrt, lambda np, x: np.sqrt(x))

        @staticmethod
        def transpose(a):
            if len(Neurality.Math.shape(a)) != 2:
                raise ValueError("transpose requires a matrix")
            if Neurality.Math.is_numpy(a):
                return a.T.copy()
            return [list(column) for column in zip(*a)]

        @staticmethod
        def matmul(a, b):
            sa, sb = Neurality.Math.shape(a), Neurality.Math.shape(b)
            if len(sa) != 2 or len(sb) != 2 or sa[1] != sb[0]:
                raise ValueError("Incompatible matrix multiplication shapes")
            if Neurality.Math.is_numpy(a) or Neurality.Math.is_numpy(b):
                return Neurality.Math.numpy().asarray(a) @ Neurality.Math.numpy().asarray(b)
            columns = Neurality.Math.transpose(b)
            return [[sum(x * y for x, y in zip(row, column)) for column in columns] for row in a]

        @staticmethod
        def column(a, index):
            if Neurality.Math.is_numpy(a):
                return a[:, index].copy()
            return [row[index] for row in a]

        @staticmethod
        def stack_columns(columns):
            if any(Neurality.Math.is_numpy(column) for column in columns):
                return Neurality.Math.numpy().column_stack(columns)
            return Neurality.Math.transpose(columns)

        @staticmethod
        def add_bias(a, bias):
            if Neurality.Math.is_numpy(a):
                return a + bias
            return [Neurality.Math.add(row, bias) for row in a]

        @staticmethod
        def mean(a):
            if Neurality.Math.is_numpy(a):
                if not a.size:
                    raise ValueError("Cannot average an empty array")
                return float(a.mean())
            values = Neurality.Math.flatten(a)
            if not values:
                raise ValueError("Cannot average an empty array")
            return sum(values) / len(values)

        @staticmethod
        def randn(size, scale=1.0):
            if Neurality._default_backend is Neurality.Backends.NUMPY:
                return Neurality.Math.numpy().random.randn(size) * scale
            return [random.gauss(0, scale) for _ in range(size)]

    class Optimizers:
        class SGD:
            def __init__(self, learning_rate=0.01):
                if not math.isfinite(learning_rate) or learning_rate <= 0:
                    raise ValueError("learning_rate must be positive and finite")
                self.learning_rate = float(learning_rate)
                self.state = {}
                self.steps = 0

            def reset(self):
                self.state.clear()
                self.steps = 0

            def step(self, parameters, learning_rate=None):
                rate = self.learning_rate if learning_rate is None else learning_rate
                if not math.isfinite(rate) or rate <= 0:
                    raise ValueError("learning_rate must be positive and finite")
                self.steps += 1
                for key, neuron, field, gradient in parameters:
                    neuron[field] = Neurality.Math.sub(neuron[field], Neurality.Math.mul(rate, self._direction(key, gradient)))

            def _direction(self, key, gradient):
                return gradient

        class Momentum(SGD):
            def __init__(self, learning_rate=0.01, momentum=0.9):
                super().__init__(learning_rate)
                if not 0 <= momentum < 1:
                    raise ValueError("momentum must be in [0, 1)")
                self.momentum = momentum

            def _direction(self, key, gradient):
                velocity = Neurality.Math.add(Neurality.Math.mul(self.momentum, self.state.get(key, 0)), gradient)
                self.state[key] = velocity
                return velocity

        class RMSprop(SGD):
            def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-8):
                super().__init__(learning_rate)
                if not 0 <= rho < 1:
                    raise ValueError("rho must be in [0, 1)")
                if not math.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("epsilon must be positive and finite")
                self.rho = rho
                self.epsilon = epsilon

            def _direction(self, key, gradient):
                square = Neurality.Math.add(Neurality.Math.mul(self.rho, self.state.get(key, 0)), Neurality.Math.mul(1 - self.rho, Neurality.Math.square(gradient)))
                self.state[key] = square
                return Neurality.Math.div(gradient, Neurality.Math.add(Neurality.Math.sqrt(square), self.epsilon))

        class Adam(SGD):
            def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                         epsilon=1e-8):
                super().__init__(learning_rate)
                if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
                    raise ValueError("beta1 and beta2 must be in [0, 1)")
                if not math.isfinite(epsilon) or epsilon <= 0:
                    raise ValueError("epsilon must be positive and finite")
                self.beta1 = beta1
                self.beta2 = beta2
                self.epsilon = epsilon

            def _direction(self, key, gradient):
                first, second = self.state.get(key, (0, 0))
                first = Neurality.Math.add(Neurality.Math.mul(self.beta1, first), Neurality.Math.mul(1 - self.beta1, gradient))
                second = Neurality.Math.add(Neurality.Math.mul(self.beta2, second), Neurality.Math.mul(1 - self.beta2, Neurality.Math.square(gradient)))
                self.state[key] = (first, second)
                corrected_first = Neurality.Math.div(first, 1 - self.beta1 ** self.steps)
                corrected_second = Neurality.Math.div(second, 1 - self.beta2 ** self.steps)
                return Neurality.Math.div(corrected_first, Neurality.Math.add(Neurality.Math.sqrt(corrected_second), self.epsilon))

    class Loss:
        @staticmethod
        def loss_mse(a, target):
            return Neurality.Math.mean(Neurality.Math.square(Neurality.Math.sub(a, target)))

        @staticmethod
        def loss_bce(a, target):
            if Neurality.Math.is_numpy(a):
                np = Neurality.Math.numpy()
                a = np.clip(a, 1e-8, 1 - 1e-8)
                target = np.asarray(target)
                return float(-np.mean(target * np.log(a) + (1 - target) * np.log(1 - a)))
            def term(p, t):
                p = min(max(p, 1e-8), 1 - 1e-8)
                return -(t * math.log(p) + (1 - t) * math.log(1 - p))
            return Neurality.Math.mean(Neurality.Math.binary(a, Neurality.Math.tolist(target), term))

        @staticmethod
        def loss_cce(a, target):
            a = Neurality.Math.rows(a)
            target = Neurality.Math.rows(target)
            if Neurality.Math.is_numpy(a):
                np = Neurality.Math.numpy()
                return float(-np.mean(np.sum(np.asarray(target) * np.log(np.clip(a, 1e-8, 1.0)), axis=1)))
            a, target = Neurality.Math.tolist(a), Neurality.Math.tolist(target)
            terms = Neurality.Math.binary(a, target, lambda p, t: -t * math.log(min(max(p, 1e-8), 1.0)))
            return sum(Neurality.Math.flatten(terms)) / len(a)

    class Gradients:
        @staticmethod
        def gradient_mse(a, target):
            size = a.size if Neurality.Math.is_numpy(a) else len(Neurality.Math.flatten(a))
            return Neurality.Math.mul(2 / size, Neurality.Math.sub(a, target))

        @staticmethod
        def gradient_bce(a, target):
            if Neurality.Math.is_numpy(a):
                np = Neurality.Math.numpy()
                a = np.clip(a, 1e-8, 1 - 1e-8)
                target = np.asarray(target)
                return (-target / a + (1 - target) / (1 - a)) / a.size
            backend = Neurality.Backends.NUMPY if Neurality.Math.is_numpy(a) else Neurality.Backends.PYTHON
            size = len(Neurality.Math.flatten(a))
            def term(p, t):
                p = min(max(p, 1e-8), 1 - 1e-8)
                return (-t / p + (1 - t) / (1 - p)) / size
            return Neurality.Math.array(Neurality.Math.binary(Neurality.Math.tolist(a), Neurality.Math.tolist(target), term), backend)

        @staticmethod
        def gradient_cce(a, target):
            shape = Neurality.Math.shape(a)
            if len(shape) > 2:
                result = Neurality.Gradients.gradient_cce(Neurality.Math.rows(a), Neurality.Math.rows(target))
                return Neurality.Math.reshape(result, shape)
            if Neurality.Math.is_numpy(a):
                np = Neurality.Math.numpy()
                return -np.asarray(target) / np.clip(a, 1e-8, 1.0) / len(a)
            backend = Neurality.Backends.NUMPY if Neurality.Math.is_numpy(a) else Neurality.Backends.PYTHON
            result = Neurality.Math.binary(Neurality.Math.tolist(a), Neurality.Math.tolist(target), lambda p, t: -t / min(max(p, 1e-8), 1.0) / len(a))
            return Neurality.Math.array(result, backend)

    class Activations:
        @staticmethod
        def gelu(x):
            """GELU using the tanh approximation used by GPT-1."""
            scale = math.sqrt(2 / math.pi)
            return Neurality.Math.unary(
                x,
                lambda v: 0.5 * v * (1 + math.tanh(scale * (v + 0.044715 * v ** 3))),
                lambda np, a: 0.5 * a * (1 + np.tanh(scale * (a + 0.044715 * a ** 3))),
            )

        @staticmethod
        def relu(x):
            return Neurality.Math.unary(x, lambda v: max(0, v), lambda np, a: np.maximum(0, a))

        @staticmethod
        def sigmoid(x):
            if Neurality.Math.is_numpy(x):
                np = Neurality.Math.numpy()
                e = np.exp(-np.abs(x))
                return np.where(x >= 0, 1 / (1 + e), e / (1 + e))
            def activate(v):
                e = math.exp(-abs(v))
                return 1 / (1 + e) if v >= 0 else e / (1 + e)
            return Neurality.Math.unary(x, activate)

        @staticmethod
        def tanh(x):
            return Neurality.Math.unary(x, math.tanh, lambda np, a: np.tanh(a))

        @staticmethod
        def linear(x):
            return Neurality.Math.array(x, Neurality.Backends.NUMPY if Neurality.Math.is_numpy(x) else Neurality.Backends.PYTHON)

        @staticmethod
        def leaky_relu(x):
            return Neurality.Math.unary(x, lambda v: v if v > 0 else 0.01 * v, lambda np, a: np.where(a > 0, a, 0.01 * a))

        @staticmethod
        def softmax(x):
            backend = Neurality.Backends.NUMPY if Neurality.Math.is_numpy(x) else Neurality.Backends.PYTHON
            if len(Neurality.Math.shape(x)) != 2:
                raise ValueError("softmax requires a matrix")
            if Neurality.Math.is_numpy(x):
                np = Neurality.Math.numpy()
                values = np.exp(x - x.max(axis=1, keepdims=True))
                return values / values.sum(axis=1, keepdims=True)
            output = []
            for row in Neurality.Math.tolist(x):
                high = max(row)
                values = [math.exp(v - high) for v in row]
                total = sum(values)
                output.append([v / total for v in values])
            return Neurality.Math.array(output, backend)

        loss_cce = staticmethod(lambda a, target: Neurality.Loss.loss_cce(a, target))

    class Derivations:
        @staticmethod
        def deriv_gelu(x):
            scale = math.sqrt(2 / math.pi)
            def derivative(v):
                t = math.tanh(scale * (v + 0.044715 * v ** 3))
                return 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * scale * (1 + 3 * 0.044715 * v * v)
            def numpy_derivative(np, a):
                t = np.tanh(scale * (a + 0.044715 * a ** 3))
                return 0.5 * (1 + t) + 0.5 * a * (1 - t * t) * scale * (1 + 3 * 0.044715 * a * a)
            return Neurality.Math.unary(x, derivative, numpy_derivative)

        @staticmethod
        def deriv_tanh(x):
            return Neurality.Math.unary(x, lambda v: 1 - math.tanh(v) ** 2, lambda np, a: 1 - np.tanh(a) ** 2)

        @staticmethod
        def deriv_linear(x):
            return Neurality.Math.unary(x, lambda v: 1.0, lambda np, a: np.ones_like(a))

        @staticmethod
        def deriv_leaky_relu(x):
            return Neurality.Math.unary(x, lambda v: 1.0 if v > 0 else 0.01, lambda np, a: np.where(a > 0, 1.0, 0.01))

        @staticmethod
        def deriv_softmax(x):
            backend = Neurality.Backends.NUMPY if Neurality.Math.is_numpy(x) else Neurality.Backends.PYTHON
            a = Neurality.Math.tolist(Neurality.Activations.softmax(x))
            return Neurality.Math.array([[[p * ((1 if i == j else 0) - row[j]) for j in range(len(row))] for i, p in enumerate(row)] for row in a], backend)

        @staticmethod
        def deriv_sigmoid(x):
            a = Neurality.Activations.sigmoid(x)
            return Neurality.Math.mul(a, Neurality.Math.sub(1, a))

        @staticmethod
        def deriv_relu(x):
            return Neurality.Math.unary(x, lambda v: float(v > 0), lambda np, a: (a > 0).astype(float))

    class Initializations:
        @staticmethod
        def kaiming(fan_in, fan_out):
            if fan_in <= 0 or fan_out <= 0:
                raise ValueError("fan_in and fan_out must be positive")
            return Neurality.Math.randn(fan_in, math.sqrt(2 / fan_in))

        @staticmethod
        def glorot(fan_in, fan_out):
            if fan_in <= 0 or fan_out <= 0:
                raise ValueError("fan_in and fan_out must be positive")
            return Neurality.Math.randn(fan_in, math.sqrt(2 / (fan_in + fan_out)))

    def __init__(self, backend=None):
        self._backend = self._default_backend if backend is None else self.Backends(backend)
        if self._backend is self.Backends.NUMPY:
            self.Math.numpy()
        self.perceptrons = []
        self.layers = []
        self.INPUT_SIZE = 2
        self.HIDDEN_LAYER = 3
        self.OUTPUT = 1
        self._optimizer = self.Optimizers.SGD(learning_rate=0.1)

    def optimizer(self, optimizer):
        if not isinstance(optimizer, self.Optimizers.SGD):
            raise TypeError("optimizer must be a Neurality.Optimizers instance")
        self._optimizer = type(optimizer).__new__(type(optimizer))
        self._optimizer.__dict__ = optimizer.__dict__.copy()
        self._optimizer.reset()
        return self

    def backend(self, backend):
        selected = self.Backends(backend)
        if selected is self.Backends.NUMPY:
            self.Math.numpy()
        for layer in self.perceptrons:
            for neuron in layer:
                neuron["w"] = self.Math.array(neuron["w"], selected)
        for layer in self.layers:
            if layer["type"] != "dense":
                for name in self._parameter_names(layer):
                    layer[name] = self.Math.array(layer[name], selected)
        if hasattr(self, "data"):
            self.data = self.Math.array(self.data, selected)
        if getattr(self, "target", None) is not None:
            self.target = self.Math.array(self.target, selected)
        self._backend = selected
        self._optimizer.reset()
        for name in ("output", "layer_inputs", "layer_linear", "_caches"):
            self.__dict__.pop(name, None)
        return self

    def fit(self, data, target=None):
        data = self.Math.array(data, self._backend)
        shape = self.Math.shape(data)
        if len(shape) not in (2, 3) or not all(shape) or not self.Math.finite(data):
            raise ValueError("data must be a non-empty finite matrix or batch of sequences")
        self.INPUT_SIZE = shape[-1]
        self.data = data
        self.target = None if target is None else self.Math.array(target, self._backend)
        return self

    @classmethod
    def _model_functions(cls):
        names = ("relu", "gelu", "sigmoid", "tanh", "linear", "leaky_relu", "softmax")
        return ({name: getattr(cls.Activations, name) for name in names},
                {f"deriv_{name}": getattr(cls.Derivations, f"deriv_{name}")
                 for name in names})

    @staticmethod
    def _parameter_names(layer):
        if layer["type"] == "embedding":
            return ("weights",)
        if layer["type"] == "attention":
            return ("Wq", "Wk", "Wv")
        if layer["type"] == "multihead_attention":
            return ("Wq", "Wk", "Wv", "Wo")
        if layer["type"] == "positional_embedding":
            return ("weights",)
        if layer["type"] == "norm":
            return ("gamma", "beta")
        return ()

    def _feature_size(self):
        if not self.layers:
            return self.INPUT_SIZE
        layer = self.layers[-1]
        return len(layer["neurons"]) if layer["type"] == "dense" else layer["embedding_size"]

    def _random_matrix(self, rows, columns, scale):
        if self._backend is self.Backends.NUMPY:
            return self.Math.numpy().random.randn(rows, columns) * scale
        return [[random.gauss(0, scale) for _ in range(columns)] for _ in range(rows)]

    def embedding(self, vocab_size, embedding_size):
        if self.layers:
            raise ValueError("embedding must be the first layer")
        if any(type(size) is not int or size <= 0 for size in (vocab_size, embedding_size)):
            raise ValueError("vocab_size and embedding_size must be positive integers")
        self.layers.append({"type": "embedding", "vocab_size": vocab_size,
                            "embedding_size": embedding_size,
                            "weights": self._random_matrix(vocab_size, embedding_size, 1 / math.sqrt(embedding_size))})
        return self

    def positional_embedding(self, max_length):
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        size = self._feature_size()
        self.layers.append({"type": "positional_embedding", "embedding_size": size,
                            "max_length": max_length,
                            "weights": self._random_matrix(max_length, size, 1 / math.sqrt(size))})
        return self

    def norm(self, epsilon=1e-5):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        size = self._feature_size()
        self.layers.append({"type": "norm", "embedding_size": size, "epsilon": float(epsilon),
                            "gamma": self.Math.array([1.0] * size, self._backend),
                            "beta": self.Math.array([0.0] * size, self._backend)})
        return self

    def residual_start(self):
        self.layers.append({"type": "residual_start", "embedding_size": self._feature_size()})
        return self

    def residual_end(self):
        depth = 0
        for layer in reversed(self.layers):
            if layer["type"] == "residual_end":
                depth += 1
            elif layer["type"] == "residual_start":
                if depth == 0:
                    if layer["embedding_size"] != self._feature_size():
                        raise ValueError("residual input and output feature sizes must match")
                    self.layers.append({"type": "residual_end", "embedding_size": self._feature_size()})
                    return self
                depth -= 1
        raise ValueError("residual_end requires a matching residual_start")

    def _validate_residuals(self):
        depth = 0
        for layer in self.layers:
            if layer["type"] == "residual_start":
                depth += 1
            elif layer["type"] == "residual_end":
                depth -= 1
                if depth < 0:
                    raise ValueError("unmatched residual_end")
        if depth:
            raise ValueError("residual_start requires a matching residual_end")

    def attention(self, key_size=None, causal=False, embedding_size=None):
        if embedding_size is not None:
            if type(embedding_size) is not int or embedding_size <= 0:
                raise ValueError("embedding_size must be a positive integer")
            if self.layers and embedding_size != self._feature_size():
                raise ValueError("embedding_size must match the previous layer output")
            if not self.layers:
                self.INPUT_SIZE = embedding_size
        size = self._feature_size()
        key_size = size if key_size is None else key_size
        if type(key_size) is not int or key_size <= 0:
            raise ValueError("key_size must be a positive integer")
        if type(causal) is not bool:
            raise ValueError("causal must be a boolean")
        self.layers.append({"type": "attention", "embedding_size": size,
                            "key_size": key_size, "causal": causal,
                            "Wq": self._random_matrix(size, key_size, math.sqrt(2 / (size + key_size))),
                            "Wk": self._random_matrix(size, key_size, math.sqrt(2 / (size + key_size))),
                            "Wv": self._random_matrix(size, size, 1 / math.sqrt(size))})
        return self

    def multihead_attention(self, num_heads, causal=False, embedding_size=None):
        if embedding_size is not None:
            if type(embedding_size) is not int or embedding_size <= 0:
                raise ValueError("embedding_size must be a positive integer")
            if self.layers and embedding_size != self._feature_size():
                raise ValueError("embedding_size must match the previous layer output")
            if not self.layers:
                self.INPUT_SIZE = embedding_size
        size = self._feature_size()
        if type(num_heads) is not int or num_heads <= 0 or size % num_heads:
            raise ValueError("num_heads must be positive and divide the feature size")
        if type(causal) is not bool:
            raise ValueError("causal must be a boolean")
        layer = {"type": "multihead_attention", "embedding_size": size,
                 "num_heads": num_heads, "causal": causal}
        for name in ("Wq", "Wk", "Wv", "Wo"):
            layer[name] = self._random_matrix(size, size, 1 / math.sqrt(size))
        self.layers.append(layer)
        return self

    def save(self, path):
        self._validate_residuals()
        activations, derivatives = self._model_functions()
        layers = []
        for layer in self.layers:
            if layer["type"] == "dense":
                neurons = []
                for neuron in layer["neurons"]:
                    activation = next((name for name, fn in activations.items() if fn is neuron["f"]), None)
                    derivative = next((name for name, fn in derivatives.items() if fn is neuron["df"]), None)
                    if activation is None or derivative is None:
                        raise ValueError("save only supports built-in Neurality activations and derivatives")
                    neurons.append({"w": self.Math.tolist(neuron["w"]), "b": float(neuron["b"]),
                                    "activation": activation, "derivative": derivative})
                layers.append({"type": "dense", "neurons": neurons})
            else:
                layers.append({name: self.Math.tolist(value) if name in self._parameter_names(layer) else value
                               for name, value in layer.items()})
        payload = {"format": "neurality", "version": 2, "input_size": self.INPUT_SIZE, "layers": layers}
        self._from_model_dict(payload, backend=self._backend)
        serialized = json.dumps(payload, indent=2, allow_nan=False)
        with open(path, "w", encoding="utf-8") as file:
            file.write(serialized)
        return self

    @classmethod
    def load(cls, path, backend=None):
        with open(path, encoding="utf-8") as file:
            return cls._from_model_dict(json.load(file), backend=backend)

    @classmethod
    def _from_model_dict(cls, payload, backend=None):
        try:
            if (not isinstance(payload, dict) or payload.get("format") != "neurality"
                    or type(payload.get("version")) is not int or payload["version"] not in (1, 2)):
                raise ValueError("unsupported model format or version")
            if type(payload["input_size"]) is not int or payload["input_size"] <= 0:
                raise ValueError("input_size must be a positive integer")
            if not isinstance(payload["layers"], list):
                raise ValueError("layers must be a list")
            model = cls(backend=backend)
            model.INPUT_SIZE = payload["input_size"]
            activations, derivatives = cls._model_functions()
            for i, saved in enumerate(payload["layers"]):
                if payload["version"] == 1:
                    saved = {"type": "dense", "neurons": saved}
                kind = saved["type"]
                size = model._feature_size()
                if kind == "dense":
                    if not isinstance(saved["neurons"], list) or not saved["neurons"]:
                        raise ValueError("each dense layer must contain neurons")
                    neurons = []
                    for j, neuron in enumerate(saved["neurons"]):
                        weights = cls.Math.array(neuron["w"], model._backend)
                        bias = float(neuron["b"])
                        if cls.Math.shape(weights) != (size,) or not cls.Math.finite(weights) or not math.isfinite(bias):
                            raise ValueError("invalid model weights or bias")
                        neurons.append({"w": weights, "b": bias, "f": activations[neuron["activation"]],
                                        "df": derivatives[neuron["derivative"]], "id": f"P-{i}-{j}"})
                    if any(p["f"] is not neurons[0]["f"] for p in neurons):
                        raise ValueError("all neurons in a layer must use the same activation")
                    model.perceptrons.append(neurons)
                    model.layers.append({"type": kind, "neurons": neurons})
                elif kind in ("residual_start", "residual_end"):
                    if saved["embedding_size"] != size:
                        raise ValueError("invalid residual feature size")
                    getattr(model, kind)()
                elif kind in ("embedding", "attention", "multihead_attention", "positional_embedding", "norm"):
                    dimension = saved["embedding_size"]
                    if type(dimension) is not int or dimension <= 0:
                        raise ValueError("embedding_size must be a positive integer")
                    layer = {"type": kind, "embedding_size": dimension}
                    if kind == "embedding":
                        vocab = saved["vocab_size"]
                        if model.layers or type(vocab) is not int or vocab <= 0:
                            raise ValueError("invalid embedding configuration")
                        layer["vocab_size"] = vocab
                        shapes = {"weights": (vocab, dimension)}
                    elif kind == "attention":
                        key = saved["key_size"]
                        if dimension != size or type(key) is not int or key <= 0 or type(saved["causal"]) is not bool:
                            raise ValueError("invalid attention configuration")
                        layer.update(key_size=key, causal=saved["causal"])
                        shapes = {"Wq": (dimension, key), "Wk": (dimension, key), "Wv": (dimension, dimension)}
                    elif kind == "multihead_attention":
                        heads = saved["num_heads"]
                        if dimension != size or type(heads) is not int or heads <= 0 or dimension % heads or type(saved["causal"]) is not bool:
                            raise ValueError("invalid multihead attention configuration")
                        layer.update(num_heads=heads, causal=saved["causal"])
                        shapes = {name: (dimension, dimension) for name in ("Wq", "Wk", "Wv", "Wo")}
                    elif kind == "positional_embedding":
                        length = saved["max_length"]
                        if dimension != size or type(length) is not int or length <= 0:
                            raise ValueError("invalid positional embedding configuration")
                        layer["max_length"] = length
                        shapes = {"weights": (length, dimension)}
                    else:
                        epsilon = saved["epsilon"]
                        if dimension != size or isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
                            raise ValueError("invalid normalization configuration")
                        layer["epsilon"] = float(epsilon)
                        shapes = {"gamma": (dimension,), "beta": (dimension,)}
                    for name, shape in shapes.items():
                        value = cls.Math.array(saved[name], model._backend)
                        if cls.Math.shape(value) != shape or not cls.Math.finite(value):
                            raise ValueError("invalid layer parameter shape or values")
                        layer[name] = value
                    model.layers.append(layer)
                else:
                    raise ValueError("unsupported layer type")
            model._validate_residuals()
            return model
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid Neurality model file: {error}") from error

    def dense(self, number, activation_function, derivative=None, bias=0,
                   initialization=None):
        if initialization is not None and not callable(initialization):
            raise TypeError("initialization must be callable or None")
        if type(number) is not int or number <= 0:
            raise ValueError("number must be a positive integer")
        if not math.isfinite(bias):
            raise ValueError("bias must be finite")
        input_size = self._feature_size()
        layer_id = len(self.perceptrons)
        if derivative is None:
            derivative = getattr(
                self.Derivations,
                f"deriv_{activation_function.__name__}"
            )

        layer = []
        for x in range(number):
            if initialization is None:
                weights = ([random.gauss(0, 0.1) for _ in range(input_size)]
                           if self._backend is self.Backends.PYTHON
                           else self.Math.numpy().random.randn(input_size) * 0.1)
            elif initialization in (self.Initializations.kaiming, self.Initializations.glorot):
                scale = math.sqrt(2 / (input_size if initialization is self.Initializations.kaiming else input_size + number))
                weights = ([random.gauss(0, scale) for _ in range(input_size)]
                           if self._backend is self.Backends.PYTHON
                           else self.Math.numpy().random.randn(input_size) * scale)
            else:
                weights = initialization(input_size, number)
            weights = self.Math.array(weights, self._backend)
            if self.Math.shape(weights) != (input_size,) or not self.Math.finite(weights):
                raise ValueError("initialization must return finite weights with shape (fan_in,)")
            layer.append({
                "w": weights,
                "b": float(bias),
                "f": activation_function,
                "df": derivative,
                "id": f"P-{layer_id}-{x}"
            })
        self.perceptrons.append(layer)
        self.layers.append({"type": "dense", "neurons": layer})
        return self

    def _restore_rows(self, rows, shape):
        return self.Math.reshape(rows, tuple(shape[:-1]) + (self.Math.shape(rows)[-1],))

    def forward(self, data, padding_mask=None):
        self._validate_residuals()
        output = self.Math.array(data, self._backend)
        if len(self.Math.shape(output)) == 1:
            output = self.Math.array([self.Math.tolist(output)], self._backend)
        shape = self.Math.shape(output)
        embedding_input = bool(self.layers and self.layers[0]["type"] == "embedding")
        if (len(shape) not in (2, 3) or not all(shape) or not self.Math.finite(output)
                or (embedding_input and len(shape) != 2)
                or (not embedding_input and shape[-1] != self.INPUT_SIZE)):
            raise ValueError("data must be finite and match the model input shape")
        masks = None
        if padding_mask is not None:
            mask = self.Math.tolist(padding_mask)
            mask_shape = self.Math.shape(mask)
            batch = shape[0] if embedding_input or len(shape) == 3 else 1
            length = shape[1] if embedding_input else shape[-2]
            if batch == 1 and mask_shape == (length,):
                mask = [mask]
                mask_shape = (1, length)
            if mask_shape != (batch, length) or any(v not in (0, 1) for v in self.Math.flatten(mask)):
                raise ValueError("padding_mask must contain 0 or 1 with shape (batch, sequence)")
            masks = mask
        self.layer_inputs, self.layer_linear, self._caches = [], [], []
        residuals = []
        for layer in self.layers:
            self.layer_inputs.append(output)
            shape = self.Math.shape(output)
            kind = layer["type"]
            if kind == "embedding":
                ids = self.Math.tolist(output)
                if any(v != int(v) or not 0 <= v < layer["vocab_size"] for v in self.Math.flatten(ids)):
                    raise ValueError("token IDs must be integers in [0, vocab_size)")
                ids = [[int(v) for v in row] for row in ids]
                table = layer["weights"]
                output = self.Math.array([[self.Math.tolist(table[token]) for token in row] for row in ids], self._backend)
                cache = {"ids": ids}
                linear = None
            elif kind == "positional_embedding":
                length = shape[-2]
                if length > layer["max_length"]:
                    raise ValueError("sequence exceeds positional embedding max_length")
                sequences = [output] if len(shape) == 2 else output
                result = [[self.Math.add(row, layer["weights"][pos]) for pos, row in enumerate(sequence)] for sequence in sequences]
                output = self.Math.array(result[0] if len(shape) == 2 else result, self._backend)
                cache, linear = {"shape": shape}, None
            elif kind == "norm":
                rows = self.Math.tolist(self.Math.rows(output))
                normalized, inverses = [], []
                for row in rows:
                    mean = sum(row) / len(row)
                    inverse = 1 / math.sqrt(sum((v - mean) ** 2 for v in row) / len(row) + layer["epsilon"])
                    normalized.append([(v - mean) * inverse for v in row])
                    inverses.append(inverse)
                result = [self.Math.add(self.Math.mul(row, layer["gamma"]), layer["beta"]) for row in normalized]
                output = self._restore_rows(self.Math.array(result, self._backend), shape)
                cache, linear = {"shape": shape, "normalized": normalized, "inverses": inverses}, None
            elif kind == "residual_start":
                residuals.append(output)
                cache, linear = {}, None
            elif kind == "residual_end":
                skip = residuals.pop()
                if self.Math.shape(skip) != shape:
                    raise ValueError("residual input and output shapes must match")
                output = self.Math.add(output, skip)
                cache, linear = {}, None
            elif kind == "attention":
                sequences = [output] if len(shape) == 2 else [output[k] for k in range(shape[0])]
                results, caches = [], []
                for sequence_index, x in enumerate(sequences):
                    q = self.Math.matmul(x, layer["Wq"])
                    k = self.Math.matmul(x, layer["Wk"])
                    v = self.Math.matmul(x, layer["Wv"])
                    scores = self.Math.div(self.Math.matmul(q, self.Math.transpose(k)), math.sqrt(layer["key_size"]))
                    probabilities = self._attention_probabilities(scores, layer["causal"], None if masks is None else masks[sequence_index])
                    results.append(self.Math.matmul(probabilities, v))
                    caches.append((x, q, k, v, probabilities))
                output = results[0] if len(shape) == 2 else self.Math.array(results, self._backend)
                cache, linear = {"sequences": caches, "shape": shape}, None
            elif kind == "multihead_attention":
                sequences = [output] if len(shape) == 2 else [output[k] for k in range(shape[0])]
                width = layer["embedding_size"] // layer["num_heads"]
                results, caches = [], []
                for sequence_index, x in enumerate(sequences):
                    projections = [self.Math.tolist(self.Math.matmul(x, layer[name])) for name in ("Wq", "Wk", "Wv")]
                    head_outputs, head_caches = [], []
                    for head in range(layer["num_heads"]):
                        start = head * width
                        q, k, v = [self.Math.array([row[start:start + width] for row in projection], self._backend) for projection in projections]
                        scores = self.Math.div(self.Math.matmul(q, self.Math.transpose(k)), math.sqrt(width))
                        probabilities = self._attention_probabilities(scores, layer["causal"], None if masks is None else masks[sequence_index])
                        head_outputs.append(self.Math.tolist(self.Math.matmul(probabilities, v)))
                        head_caches.append((q, k, v, probabilities))
                    combined = self.Math.array([[value for head in head_outputs for value in head[row]] for row in range(shape[-2])], self._backend)
                    results.append(self.Math.matmul(combined, layer["Wo"]))
                    caches.append((x, combined, head_caches))
                output = results[0] if len(shape) == 2 else self.Math.array(results, self._backend)
                cache, linear = {"sequences": caches, "shape": shape}, None
            else:
                x = self.Math.rows(output)
                neurons = layer["neurons"]
                weights = self.Math.stack_columns([p["w"] for p in neurons])
                linear = self.Math.add_bias(self.Math.matmul(x, weights), [p["b"] for p in neurons])
                result = self.Math.array(neurons[0]["f"](linear), self._backend)
                output = self._restore_rows(result, shape)
                cache = {"x": x, "shape": shape, "linear": linear}
            self.layer_linear.append(linear)
            self._caches.append(cache)
        self.output = output
        return output

    def backward(self, target, gradient_function, learning_rate=None):
        if not hasattr(self, "output") or not hasattr(self, "_caches"):
            raise ValueError("Call forward before backward")
        shape = self.Math.shape(self.output)
        if len(shape) < 2 or len(self.Math.flatten(target)) != math.prod(shape):
            raise ValueError("target must match the output shape")
        target = self.Math.reshape(self.Math.array(target, self._backend), shape)
        prediction_rows, target_rows = self.Math.rows(self.output), self.Math.rows(target)
        gradient = self.Math.array(gradient_function(prediction_rows, target_rows), self._backend)
        if self.Math.shape(gradient) != self.Math.shape(prediction_rows):
            raise ValueError("gradient_function must return the output shape")
        gradient = self.Math.reshape(gradient, shape)
        if (hasattr(self, "data") and self.layer_inputs
                and self.Math.tolist(self.layer_inputs[0]) == self.Math.tolist(self.data)):
            self.target = self.Math.array(target, self._backend)
        parameters = []
        residual_gradients = []
        for i in reversed(range(len(self.layers))):
            layer, cache = self.layers[i], self._caches[i]
            if layer["type"] == "dense":
                neurons = layer["neurons"]
                gradient = self.Math.rows(gradient)
                weights = self.Math.stack_columns([p["w"] for p in neurons])
                if (neurons[0]["f"] is self.Activations.softmax
                        and neurons[0]["df"] is self.Derivations.deriv_softmax):
                    probabilities = self.Math.tolist(self.Activations.softmax(cache["linear"]))
                    if i == len(self.layers) - 1 and gradient_function is self.Gradients.gradient_cce:
                        targets = self.Math.tolist(target_rows)
                        delta = [[(p * sum(t) - v) / len(targets) for p, v in zip(row, t)]
                                 for row, t in zip(probabilities, targets)]
                    else:
                        delta = self._softmax_product(probabilities, self.Math.tolist(gradient))
                    delta = self.Math.array(delta, self._backend)
                else:
                    derivative = self.Math.stack_columns([p["df"](self.Math.column(cache["linear"], j)) for j, p in enumerate(neurons)])
                    delta = self.Math.mul(gradient, derivative)
                dw = self.Math.matmul(self.Math.transpose(cache["x"]), delta)
                gradient = self._restore_rows(self.Math.matmul(delta, self.Math.transpose(weights)), cache["shape"])
                for j, neuron in enumerate(neurons):
                    parameters.append(((i, j, "w"), neuron, "w", self.Math.column(dw, j)))
                    parameters.append(((i, j, "b"), neuron, "b", sum(self.Math.column(delta, j))))
            elif layer["type"] == "residual_end":
                residual_gradients.append(gradient)
            elif layer["type"] == "residual_start":
                gradient = self.Math.add(gradient, residual_gradients.pop())
            elif layer["type"] == "positional_embedding":
                shape = cache["shape"]
                sequences = [self.Math.tolist(gradient)] if len(shape) == 2 else self.Math.tolist(gradient)
                dw = [[0.0] * layer["embedding_size"] for _ in range(layer["max_length"])]
                for sequence in sequences:
                    for pos, row in enumerate(sequence):
                        dw[pos] = self.Math.add(dw[pos], row)
                parameters.append(((i, "weights"), layer, "weights", self.Math.array(dw, self._backend)))
            elif layer["type"] == "norm":
                rows = self.Math.tolist(self.Math.rows(gradient))
                gamma = self.Math.tolist(layer["gamma"])
                width = layer["embedding_size"]
                dgamma, dbeta = [0.0] * width, [0.0] * width
                inputs = []
                for dy, xhat, inverse in zip(rows, cache["normalized"], cache["inverses"]):
                    scaled = [d * g for d, g in zip(dy, gamma)]
                    avg = sum(scaled) / width
                    avg_x = sum(d * x for d, x in zip(scaled, xhat)) / width
                    inputs.append([inverse * (d - avg - x * avg_x) for d, x in zip(scaled, xhat)])
                    dgamma = [a + d * x for a, d, x in zip(dgamma, dy, xhat)]
                    dbeta = [a + d for a, d in zip(dbeta, dy)]
                gradient = self._restore_rows(self.Math.array(inputs, self._backend), cache["shape"])
                for name, value in (("gamma", dgamma), ("beta", dbeta)):
                    parameters.append(((i, name), layer, name, self.Math.array(value, self._backend)))
            elif layer["type"] == "multihead_attention":
                sequence_gradients = [gradient] if len(cache["shape"]) == 2 else [gradient[k] for k in range(cache["shape"][0])]
                accumulated = {name: self.Math.mul(layer[name], 0) for name in self._parameter_names(layer)}
                width = layer["embedding_size"] // layer["num_heads"]
                inputs = []
                for (x, combined, heads), dy in zip(cache["sequences"], sequence_gradients):
                    accumulated["Wo"] = self.Math.add(accumulated["Wo"], self.Math.matmul(self.Math.transpose(combined), dy))
                    dcombined = self.Math.tolist(self.Math.matmul(dy, self.Math.transpose(layer["Wo"])))
                    head_derivatives = []
                    for head, (q, k, v, probabilities) in enumerate(heads):
                        dh = self.Math.array([row[head * width:(head + 1) * width] for row in dcombined], self._backend)
                        dv = self.Math.matmul(self.Math.transpose(probabilities), dh)
                        dp = self.Math.matmul(dh, self.Math.transpose(v))
                        ds = self.Math.div(self.Math.array(self._softmax_product(self.Math.tolist(probabilities), self.Math.tolist(dp)), self._backend), math.sqrt(width))
                        head_derivatives.append([self.Math.tolist(value) for value in (self.Math.matmul(ds, k), self.Math.matmul(self.Math.transpose(ds), q), dv)])
                    dx = self.Math.mul(x, 0)
                    for index, name in enumerate(("Wq", "Wk", "Wv")):
                        projected = self.Math.array([[value for head in head_derivatives for value in head[index][row]] for row in range(len(dcombined))], self._backend)
                        accumulated[name] = self.Math.add(accumulated[name], self.Math.matmul(self.Math.transpose(x), projected))
                        dx = self.Math.add(dx, self.Math.matmul(projected, self.Math.transpose(layer[name])))
                    inputs.append(dx)
                for name, value in accumulated.items():
                    parameters.append(((i, name), layer, name, value))
                gradient = inputs[0] if len(cache["shape"]) == 2 else self.Math.array(inputs, self._backend)
            elif layer["type"] == "attention":
                sequence_gradients = [gradient] if len(cache["shape"]) == 2 else [gradient[k] for k in range(cache["shape"][0])]
                accumulated = {name: self.Math.mul(layer[name], 0) for name in self._parameter_names(layer)}
                inputs = []
                for (x, q, k, v, probabilities), dy in zip(cache["sequences"], sequence_gradients):
                    dv = self.Math.matmul(self.Math.transpose(probabilities), dy)
                    dp = self.Math.matmul(dy, self.Math.transpose(v))
                    ds = self.Math.array(self._softmax_product(self.Math.tolist(probabilities), self.Math.tolist(dp)), self._backend)
                    ds = self.Math.div(ds, math.sqrt(layer["key_size"]))
                    dq = self.Math.matmul(ds, k)
                    dk = self.Math.matmul(self.Math.transpose(ds), q)
                    xt = self.Math.transpose(x)
                    for name, value in (("Wq", dq), ("Wk", dk), ("Wv", dv)):
                        accumulated[name] = self.Math.add(accumulated[name], self.Math.matmul(xt, value))
                    dx = self.Math.add(self.Math.matmul(dq, self.Math.transpose(layer["Wq"])), self.Math.matmul(dk, self.Math.transpose(layer["Wk"])))
                    inputs.append(self.Math.add(dx, self.Math.matmul(dv, self.Math.transpose(layer["Wv"]))))
                for name, value in accumulated.items():
                    parameters.append(((i, name), layer, name, value))
                gradient = inputs[0] if len(cache["shape"]) == 2 else self.Math.array(inputs, self._backend)
            else:
                dw = [[0.0] * layer["embedding_size"] for _ in range(layer["vocab_size"])]
                for tokens, sequence in zip(cache["ids"], self.Math.tolist(gradient)):
                    for token, values in zip(tokens, sequence):
                        dw[token] = self.Math.add(dw[token], values)
                parameters.append(((i, "weights"), layer, "weights", self.Math.array(dw, self._backend)))
        self._optimizer.step(parameters, learning_rate=learning_rate)
        return self

    @staticmethod
    def _attention_probabilities(scores, causal, mask):
        rows = Neurality.Math.tolist(scores)
        result = []
        for row_index, row in enumerate(rows):
            allowed = [j for j in range(len(row)) if (mask is None or mask[j]) and (not causal or j <= row_index)]
            if not allowed:
                result.append([0.0] * len(row))
                continue
            high = max(row[j] for j in allowed)
            allowed = set(allowed)
            values = [math.exp(row[j] - high) if j in allowed else 0.0 for j in range(len(row))]
            total = sum(values)
            result.append([value / total for value in values])
        backend = Neurality.Backends.NUMPY if Neurality.Math.is_numpy(scores) else Neurality.Backends.PYTHON
        return Neurality.Math.array(result, backend)

    @staticmethod
    def _softmax_product(probabilities, gradients):
        result = []
        for row, gradient in zip(probabilities, gradients):
            dot = sum(p * g for p, g in zip(row, gradient))
            result.append([p * (g - dot) for p, g in zip(row, gradient)])
        return result




