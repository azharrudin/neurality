import tempfile
import unittest
from pathlib import Path

from lib.neurality import Neurality as N


class MultiheadTests(unittest.TestCase):
    def test_gradients_mask_and_save(self):
        x = [[0, 1, 2], [2, 1, 0]]
        mask = [[1, 1, 0], [0, 1, 1]]
        target = [[[0.1, -0.2], [0.2, 0.3], [-0.1, 0.2]]] * 2
        for backend in N.Backends:
            model = N(backend=N.Backends.PYTHON).embedding(3, 4).multihead_attention(2, causal=True)
            model.dense(2, N.Activations.linear).backend(backend)
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
                            plus = N.Loss.loss_mse(model.forward(x, padding_mask=mask), target)
                            row[index] = old - 1e-6
                            minus = N.Loss.loss_mse(model.forward(x, padding_mask=mask), target)
                            row[index] = old
                            expected.append((owner, name, a, b, old, (plus - minus) / 2e-6))
            model.forward(x, padding_mask=mask)
            model.backward(target, N.Gradients.gradient_mse, learning_rate=0.1)
            for owner, name, a, b, old, numerical in expected:
                actual = owner[name][a] if b is None else owner[name][a][b]
                self.assertAlmostEqual((old - actual) / 0.1, numerical, places=6)
            prediction = model.forward(x, padding_mask=mask)
            for q, k, v, probabilities in model._caches[1]['sequences'][1][2]:
                self.assertEqual(N.Math.tolist(probabilities)[0], [0.0, 0.0, 0.0])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.json'
                model.save(path)
                for other in N.Backends:
                    restored = N.load(path, backend=other)
                    for a, b in zip(N.Math.flatten(prediction), N.Math.flatten(restored.forward(x, padding_mask=mask))):
                        self.assertAlmostEqual(a, b, places=10)

    def test_mask_behavior_and_validation(self):
        for backend in N.Backends:
            for multi in (False, True):
                model = N(backend=backend)
                if multi:
                    model.multihead_attention(2, embedding_size=4)
                else:
                    model.attention(embedding_size=4)
                x = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
                changed = [x[0], x[1], [40, 30, 20, 10]]
                original = model.forward(x, padding_mask=[1, 1, 0])
                modified = model.forward(changed, padding_mask=[1, 1, 0])
                for a, b in zip(N.Math.flatten(original[:2]), N.Math.flatten(modified[:2])):
                    self.assertAlmostEqual(a, b)
                for invalid in ([1, 0], [1, 2, 0], [[1, 0, 1], [1, 1, 1]]):
                    with self.assertRaises(ValueError):
                        model.forward(x, padding_mask=invalid)
                self.assertTrue(N.Math.finite(model.forward(x, padding_mask=[0, 0, 0])))
            for heads in (0, 3, True):
                with self.assertRaises(ValueError):
                    N(backend=backend).multihead_attention(heads, embedding_size=4)


if __name__ == '__main__':
    unittest.main()
