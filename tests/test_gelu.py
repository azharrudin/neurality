import tempfile
import unittest
from pathlib import Path

from lib.neurality import Neurality as N


class GeluTests(unittest.TestCase):
    def test_values_and_numerical_derivative(self):
        values = [-10, -2, -1, 0, 1, 2, 10]
        for backend in N.Backends:
            with self.subTest(backend=backend):
                x = N.Math.array([values], backend)
                output = N.Math.flatten(N.Activations.gelu(x))
                self.assertAlmostEqual(output[2], -0.1588080094, places=9)
                self.assertEqual(output[3], 0)
                self.assertAlmostEqual(output[4], 0.8411919906, places=9)
                derivatives = N.Math.flatten(N.Derivations.deriv_gelu(x))
                for value, derivative in zip(values, derivatives):
                    plus = N.Activations.gelu(N.Math.array([[value + 1e-6]], backend))
                    minus = N.Activations.gelu(N.Math.array([[value - 1e-6]], backend))
                    numerical = (N.Math.flatten(plus)[0] - N.Math.flatten(minus)[0]) / 2e-6
                    self.assertAlmostEqual(derivative, numerical, places=8)

    def test_dense_backward_and_roundtrip(self):
        for backend in N.Backends:
            with self.subTest(backend=backend):
                model = N(backend=backend).fit([[1.0]]).dense(1, N.Activations.gelu)
                neuron = model.layers[0]['neurons'][0]
                neuron['w'][0] = 0.7
                old = float(neuron['w'][0])
                losses = []
                for offset in (1e-6, -1e-6):
                    neuron['w'][0] = old + offset
                    losses.append(N.Loss.loss_mse(model.forward([[1.0]]), [[0.2]]))
                neuron['w'][0] = old
                model.forward([[1.0]])
                model.backward([[0.2]], N.Gradients.gradient_mse, learning_rate=0.1)
                self.assertAlmostEqual((old - neuron['w'][0]) / 0.1,
                                       (losses[0] - losses[1]) / 2e-6, places=7)
                expected = N.Math.flatten(model.forward([[1.0]]))[0]
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'gelu.json'
                    model.save(path)
                    for other in N.Backends:
                        loaded = N.load(path, backend=other)
                        self.assertAlmostEqual(N.Math.flatten(loaded.forward([[1.0]]))[0], expected)


if __name__ == '__main__':
    unittest.main()
