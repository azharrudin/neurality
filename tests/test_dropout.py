import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lib.neurality import Neurality as N


class DropoutTests(unittest.TestCase):
    def test_eval_is_identity(self):
        for backend in N.Backends:
            model = N(backend=backend).dropout(0.5)
            data = [[1.0, 2.0], [3.0, 4.0]]
            self.assertEqual(N.Math.tolist(model.forward(data, training=False)), data)
            model.eval()
            self.assertEqual(N.Math.tolist(model.forward(data)), data)

    def test_train_mode_masks_and_scales(self):
        for backend in N.Backends:
            np.random.seed(0)
            random.seed(0)
            data = [[1.0] * 10 for _ in range(10)]
            model = N(backend=backend).fit(data).dropout(0.9)
            values = [value for row in N.Math.tolist(model.forward(data, training=True)) for value in row]
            self.assertTrue(all(abs(value) < 1e-12 or abs(value - 10.0) < 1e-12 for value in values))
            self.assertTrue(any(abs(value) < 1e-12 for value in values))
            self.assertTrue(any(abs(value - 10.0) < 1e-12 for value in values))

            no_op = N(backend=backend).fit(data).dropout(0.0)
            self.assertEqual(N.Math.tolist(no_op.forward(data, training=True)), data)

    def test_backward_applies_mask(self):
        for backend in N.Backends:
            model = N(backend=backend)
            model.dense(1, N.Activations.linear, bias=0)
            model.dropout(0.5)
            data = [[1.0, 2.0], [3.0, 4.0]]
            target = [[0.0], [0.0]]
            model.forward(data)
            mask = N.Math.tolist(model._caches[1]["mask"])
            weights = N.Math.tolist(model.layers[0]["neurons"][0]["w"])
            dense = [sum(value * weight for value, weight in zip(row, weights)) for row in data]
            masked = [dense[row] * mask[row][0] for row in range(len(data))]
            gradient = [masked[row] * mask[row][0] for row in range(len(data))]
            expected = [
                weights[index] - sum(data[row][index] * gradient[row] for row in range(len(data)))
                for index in range(len(weights))
            ]
            model.backward(target, N.Gradients.gradient_mse, learning_rate=1.0)
            actual = N.Math.tolist(model.layers[0]["neurons"][0]["w"])
            for value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(value, expected_value, places=8)

    def test_train_eval_toggle(self):
        model = N(backend=N.Backends.PYTHON).dropout(0.5)
        self.assertTrue(model.training)
        self.assertIs(model.eval(), model)
        self.assertFalse(model.training)
        self.assertIs(model.train(), model)
        self.assertTrue(model.training)

    def test_invalid_rate(self):
        for backend in N.Backends:
            for rate in (-0.1, 1.0, float("nan"), True):
                with self.assertRaises(ValueError):
                    N(backend=backend).dropout(rate)

    def test_serialization_roundtrip(self):
        for backend in N.Backends:
            model = N(backend=backend).dropout(0.25)
            data = [[1.0, 2.0], [3.0, 4.0]]
            model.forward(data, training=False)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.json"
                model.save(path)
                for other in N.Backends:
                    loaded = N.load(path, backend=other)
                    self.assertEqual(loaded.layers[0]["type"], "dropout")
                    self.assertAlmostEqual(float(loaded.layers[0]["rate"]), 0.25)
                    loaded.eval()
                    self.assertEqual(N.Math.tolist(loaded.forward(data)), data)


if __name__ == "__main__":
    unittest.main()