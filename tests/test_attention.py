import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lib import neurality

N = neurality.Neurality


class AttentionTests(unittest.TestCase):
    def model(self, backend, causal=True, objective='mse'):
        random.seed(7)
        model = N(backend=N.Backends.PYTHON)
        model.embedding(4, 3).attention(key_size=2, causal=causal)
        model.dense(4 if objective == 'cce' else 2,
                    N.Activations.softmax if objective == 'cce' else N.Activations.linear)
        model.backend(backend)
        return model

    def assert_close(self, a, b, places=10):
        self.assertEqual(N.Math.shape(a), N.Math.shape(b))
        for x, y in zip(N.Math.flatten(a), N.Math.flatten(b)):
            self.assertAlmostEqual(x, y, places=places)

    def test_all_parameter_gradients(self):
        tokens = [[1, 1, 2], [2, 0, 1]]
        for backend in N.Backends:
            for causal in (False, True):
                model = self.model(backend, causal)
                target = [[[0.1, -0.2], [0.4, 0.1], [-0.1, 0.2]]] * 2
                expected = []
                for layer in model.layers:
                    fields = [(layer, name) for name in model._parameter_names(layer)]
                    if layer['type'] == 'dense':
                        fields += [(neuron, 'w') for neuron in layer['neurons']]
                    for owner, field in fields:
                        shape = N.Math.shape(owner[field])
                        for a in range(shape[0]):
                            indices = range(shape[1]) if len(shape) == 2 else [None]
                            for b in indices:
                                row = owner[field] if b is None else owner[field][a]
                                index = a if b is None else b
                                old = row[index]
                                row[index] = old + 1e-6
                                plus = N.Loss.loss_mse(model.forward(tokens), target)
                                row[index] = old - 1e-6
                                minus = N.Loss.loss_mse(model.forward(tokens), target)
                                row[index] = old
                                expected.append((owner, field, a, b, old, (plus - minus) / 2e-6))
                model.forward(tokens)
                model.backward(target, N.Gradients.gradient_mse, learning_rate=0.1)
                for owner, field, a, b, old, numerical in expected:
                    actual = owner[field][a] if b is None else owner[field][a][b]
                    self.assertAlmostEqual((old - actual) / 0.1, numerical, places=7)

    def test_causal_mask(self):
        for backend in N.Backends:
            model = self.model(backend)
            original = model.forward([[0, 1, 2]])
            changed = model.forward([[0, 1, 3]])
            self.assert_close(original[0][:2], changed[0][:2])
            attention = model._caches[1]['sequences'][0][-1]
            self.assertEqual(float(attention[0][1]), 0)
            self.assertEqual(float(attention[1][2]), 0)
            for row in attention:
                self.assertAlmostEqual(sum(row), 1)

    def test_backend_optimizer_parity(self):
        tokens = [[0, 1], [1, 0]]
        target = [[[0.2, 0.1], [0.4, -0.2]]] * 2
        for name in ('SGD', 'Momentum', 'RMSprop', 'Adam'):
            py, np = [self.model(backend) for backend in N.Backends]
            for model in (py, np):
                model.optimizer(getattr(N.Optimizers, name)(learning_rate=0.01))
                for _ in range(3):
                    model.forward(tokens)
                    model.backward(target, N.Gradients.gradient_mse)
            self.assert_close(py.forward(tokens), np.forward(tokens))
            self.assertEqual(py._optimizer.steps, 3)

    def test_learning_token_prediction(self):
        tokens = [[0, 1, 2], [1, 2, 0]]
        labels = [[1, 2, 0], [2, 0, 1]]
        targets = [[[float(k == label) for k in range(4)] for label in row] for row in labels]
        for backend in N.Backends:
            model = self.model(backend, objective='cce')
            model.optimizer(N.Optimizers.Adam(learning_rate=0.03))
            initial = N.Loss.loss_cce(model.forward(tokens), targets)
            for _ in range(100):
                model.forward(tokens)
                model.backward(targets, N.Gradients.gradient_cce)
            prediction = model.forward(tokens)
            final = N.Loss.loss_cce(prediction, targets)
            self.assertLess(final, initial * 0.15)
            for row, expected in zip(prediction, labels):
                actual = [max(range(len(p)), key=lambda i: p[i]) for p in row]
                self.assertEqual(actual, expected)

    def test_attention_without_embedding(self):
        for backend in N.Backends:
            model = N(backend=backend).attention(embedding_size=2, key_size=3)
            single = [[0.2, 0.3], [0.5, -0.1]]
            self.assertEqual(N.Math.shape(model.forward(single)), (2, 2))
            model.backward([[0, 0], [0, 0]], N.Gradients.gradient_mse)
            self.assertEqual(N.Math.shape(model.forward([single, single])), (2, 2, 2))
            model.backward([[[0, 0], [0, 0]]] * 2, N.Gradients.gradient_mse)

    def test_stacked_attention_gradient(self):
        for backend in N.Backends:
            model = N(backend=backend).embedding(3, 2)
            model.dense(3, N.Activations.tanh).attention(key_size=2, causal=True)
            model.attention(key_size=3).dense(3, N.Activations.softmax)
            x = [[0, 1, 0]]
            y = [[[0, 1, 0], [1, 0, 0], [0, 1, 0]]]
            neuron = model.perceptrons[0][0]
            old = float(neuron['w'][0])
            neuron['w'][0] = old + 1e-6
            plus = N.Loss.loss_cce(model.forward(x), y)
            neuron['w'][0] = old - 1e-6
            minus = N.Loss.loss_cce(model.forward(x), y)
            neuron['w'][0] = old
            model.forward(x)
            model.backward(y, N.Gradients.gradient_cce, learning_rate=0.1)
            self.assertAlmostEqual((old - neuron['w'][0]) / 0.1, (plus - minus) / 2e-6, places=7)

    def test_json_and_switch(self):
        model = self.model(N.Backends.PYTHON)
        tokens = [[1, 2, 3]]
        expected = model.forward(tokens)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            model.save(path)
            for backend in N.Backends:
                restored = N.load(path, backend=backend)
                self.assert_close(restored.forward(tokens), expected)
                restored.backend(N.Backends.PYTHON)
                self.assert_close(restored.forward(tokens), expected)
                restored.backward([[[0, 0]] * 3], N.Gradients.gradient_mse)
            payload = json.loads(path.read_text())
            payload['layers'][1]['Wq'] = [[1]]
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                N.load(path, backend=N.Backends.PYTHON)
        path = Path('examples/titanic_survivor/model.json')
        if path.exists():
            for backend in N.Backends:
                old = N.load(path, backend=backend)
                old.forward([[0] * old.INPUT_SIZE])

    def test_dense_regression_and_validation(self):
        for backend in N.Backends:
            random.seed(1)
            model = N(backend=backend).fit([[0, 1], [1, 0]])
            model.dense(3, N.Activations.tanh).dense(1, N.Activations.sigmoid)
            prediction = model.forward([[0, 1], [1, 0]])
            self.assertEqual(N.Math.shape(prediction), (2, 1))
            model.backward([[0], [1]], N.Gradients.gradient_bce)
            model = N(backend=backend).embedding(3, 2)
            for invalid in ([[0, 3]], [[-1]], [[0.5]], [[[0]]]):
                with self.assertRaises(ValueError):
                    model.forward(invalid)
            with self.assertRaises(ValueError):
                model.embedding(3, 2)
            self.assertEqual(N.Math.shape(model.forward([0, 0])), (1, 2, 2))

    def test_without_numpy(self):
        code = '''
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] == 'numpy':
        raise ImportError('NumPy blocked')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from lib.neurality import Neurality as N
m = N(backend=N.Backends.PYTHON).embedding(3, 2).attention(causal=True)
m.dense(3, N.Activations.softmax)
m.optimizer(N.Optimizers.Adam())
x = [[0, 1, 0]]
y = [[[0, 1, 0], [1, 0, 0], [0, 1, 0]]]
m.forward(x)
m.backward(y, N.Gradients.gradient_cce)
import tempfile
with tempfile.TemporaryDirectory() as directory:
    path = directory + '/model.json'
    m.save(path)
    loaded = N.load(path, backend=N.Backends.PYTHON)
    assert loaded.forward(x) == m.forward(x)
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
