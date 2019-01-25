import unittest

import numpy as np

from lorenz_systems import Lorenz63, CoupledLorenz63

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

class TestCoupledLorenz63(unittest.TestCase):
    def test_system_definition(self):
        sigma = 10.0 * np.random.rand(3)
        rho = 28.0 * np.random.rand(3)
        beta = 8.0 * np.random.rand(3) / 3.0

        ce = np.random.rand()
        c = np.random.rand()
        cz = np.random.rand()

        k1 = 10.0
        k2 = -11.0

        tau = np.random.rand()
        s = 2.0 * np.random.rand()

        model = CoupledLorenz63(sigma=sigma, rho=rho, beta=beta,
                                ce=ce, c=c, cz=cz, k1=k1, k2=k2,
                                tau=tau, s=s)

        t = np.random.rand()

        x = np.array([1.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0])
        sys = model.system(t, x)
        expected = np.array(
            [-sigma[0] - ce * k1, rho[0] + ce * k1, 0.0,
             -c * k2 - s * ce - ce * k1, c * k2 + ce * k1, 0.0,
             -c * k2, c * k2, 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0])
        sys = model.system(t, x)
        expected = np.array(
            [sigma[0] - ce * k1, -1.0 + ce * k1, 0.0,
             -c * k2 - ce * k1, c * k2 + ce * s + ce * k1, 0.0,
             -c * k2, c * k2, 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 0.0, 1.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0])
        sys = model.system(t, x)
        expected = np.array(
            [-ce * k1, ce * k1, -beta[0],
             -c * k2 - ce * k1, c * k2 + ce * k1, 0.0,
             -c * k2, c * k2, 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0,
                      0.0, 0.0, 0.0])
        sys = model.system(t, x)
        expected = np.array(
            [-ce * (s + k1), ce * k1, 0.0,
             -sigma[1] - c * k2 - ce * k1, rho[1] + c * k2 + ce * k1, 0.0,
             -c * (1.0 + k2), c * k2, 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0])
        sys = model.system(t, x)
        expected = np.array(
            [-ce * k1, ce * (s + k1), 0.0,
             sigma[1] - c * k2 - ce * k1, -1.0 + c * k2 + ce * k1, 0.0,
             -c * k2, c * (1.0 + k2), 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 0.0, 0.0,
                      0.0, 0.0, 1.0,
                      0.0, 0.0, 0.0])
        sys = model.system(t, x)
        expected = np.array(
            [-ce * k1, ce * k1, 0.0,
             -c * k2 - ce * k1, c * k2 + ce * k1, -beta[1],
             -c * k2, c * k2, -cz])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      1.0, 0.0, 0.0])
        sys = model.system(t, x)
        expected = np.array(
            [-ce * k1, ce * k1, 0.0,
             -c * (s + k2) - ce * k1, c * k2 + ce * k1, 0.0,
             -tau * sigma[2] - c * k2, tau * rho[2] + c * k2, 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0])
        sys = model.system(t, x)
        expected = np.array(
            [-ce * k1, ce * k1, 0.0,
             -c * k2 - ce * k1, c * (s + k2) + ce * k1, 0.0,
             tau * sigma[2] - c * k2, -tau + c * k2, 0.0])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0,
                      0.0, 0.0, 1.0])
        sys = model.system(t, x)
        expected = np.array(
            [-ce * k1, ce * k1, 0.0,
             -c * k2 - ce * k1, c * k2 + ce * k1, cz,
             -c * k2, c * k2, -tau * beta[2]])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))
