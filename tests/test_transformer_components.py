import tempfile
import unittest
from pathlib import Path

from lib.neurality import Neurality as N


class ComponentTests(unittest.TestCase):
    def test_gradients_and_roundtrip(self):
        for backend in N.Backends:
            model = N(backend=N.Backends.PYTHON).embedding(3, 3).positional_embedding(4)
            model.residual_start().norm().dense(3, N.Activations.tanh).residual_end()
            model.norm().dense(2, N.Activations.linear).backend(backend)
            x = [[0, 1], [1, 2]]
            target = [[[0.3, -0.1], [0.2, 0.4]], [[-0.2, 0.1], [0.5, -0.3]]]
            expected = []
            for layer in model.layers:
                fields = [(layer, name) for name in model._parameter_names(layer)]
                if layer['type'] == 'dense':
                    fields += [(neuron, 'w') for neuron in layer['neurons']]
                for owner, name in fields:
                    shape = N.Math.shape(owner[name])
                    for a in range(shape[0]):
                        for b in (range(shape[1]) if len(shape) == 2 else [None]):
                            row, index = (owner[name], a) if b is None else (owner[name][a], b)
                            old = float(row[index])
                            row[index] = old + 1e-6
                            plus = N.Loss.loss_mse(model.forward(x), target)
                            row[index] = old - 1e-6
                            minus = N.Loss.loss_mse(model.forward(x), target)
                            row[index] = old
                            expected.append((owner, name, a, b, old, (plus - minus) / 2e-6))
            model.forward(x)
            model.backward(target, N.Gradients.gradient_mse, learning_rate=0.1)
            for owner, name, a, b, old, numerical in expected:
                actual = owner[name][a] if b is None else owner[name][a][b]
                self.assertAlmostEqual((old - actual) / 0.1, numerical, places=6)
            prediction = N.Math.tolist(model.forward(x))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.json'
                model.save(path)
                for other in N.Backends:
                    loaded = N.load(path, backend=other)
                    for actual, expected_value in zip(N.Math.flatten(loaded.forward(x)), N.Math.flatten(prediction)):
                        self.assertAlmostEqual(actual, expected_value, places=10)
                    loaded.backend(backend).forward(x)
                    loaded.backward(target, N.Gradients.gradient_mse)

    def test_nested_residuals_and_validation(self):
        for backend in N.Backends:
            model = N(backend=backend).residual_start().residual_start().residual_end().residual_end()
            self.assertEqual(N.Math.tolist(model.forward([[1, 2]])), [[3, 6]])
            with self.assertRaises(ValueError):
                model.residual_end()
            incomplete = N(backend=backend).residual_start()
            with self.assertRaises(ValueError):
                incomplete.forward([[1, 2]])
            with self.assertRaises(ValueError):
                incomplete.dense(3, N.Activations.linear).residual_end()
            pos = N(backend=backend).positional_embedding(1)
            with self.assertRaises(ValueError):
                pos.forward([[1, 2], [3, 4]])
            for value in (0, -1, float('nan'), True):
                with self.assertRaises(ValueError):
                    N(backend=backend).norm(value)


if __name__ == '__main__':
    unittest.main()
