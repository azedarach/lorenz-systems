import unittest

import numpy as np

from lorenz_systems import Lorenz63

class TestLorenz63(unittest.TestCase):
    def test_system_definition(self):
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0

        model = Lorenz63(sigma=sigma, rho=rho, beta=beta)

        t = np.random.rand()

        x = np.array([1.0, 0.0, 0.0])
        sys = model.system(t, x)
        expected = np.array([-sigma, rho, 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 1.0, 0.0])
        sys = model.system(t, x)
        expected = np.array([sigma, -1.0, 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 0.0, 1.0])
        sys = model.system(t, x)
        expected = np.array([0.0, 0.0, -beta])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([1.0, 1.0, 0.0])
        sys = model.system(t, x)
        expected = np.array([0.0, rho - 1.0, 1.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([1.0, 0.0, 1.0])
        sys = model.system(t, x)
        expected = np.array([-sigma, rho - 1.0, -beta])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 1.0, 1.0])
        sys = model.system(t, x)
        expected = np.array([sigma, -1.0, -beta])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))
