import unittest

import numpy as np

from lorenz_systems import Lorenz96I, Lorenz96II

class TestLorenz96I(unittest.TestCase):
    def test_system_definition_1d(self):
        n = 1
        forcing = lambda t, x : 1

        model = Lorenz96I(n=n, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        sys = model.system(t, x)
        expected = forcing(t, x) * np.ones(n)
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.689])
        sys = model.system(t, x)
        expected = np.array([-x[0] + forcing(t, x)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

    def test_system_definition_2d(self):
        n = 2
        forcing = lambda t, x : 1

        model = Lorenz96I(n=n, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        sys = model.system(t, x)
        expected = forcing(t, x) * np.ones(n)
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([5.9993, 3.3973])
        sys = model.system(t, x)
        expected = np.array([x[1] * (x[1] - x[0]) - x[0] + forcing(t, x),
                             x[0] * (x[0] - x[1]) - x[1] + forcing(t, x)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

    def test_system_definition_3d(self):
        n = 3
        forcing = lambda t, x : -3

        model = Lorenz96I(n=n, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        sys = model.system(t, x)
        expected = forcing(t, x) * np.ones(n)
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.7058, 4.2932, 1.4289])
        sys = model.system(t, x)
        expected = np.array([-x[0] + forcing(t, x),
                             -x[1] + forcing(t, x),
                             -x[2] + forcing(t, x)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

    def test_system_definition_4d(self):
        n = 4
        forcing = lambda t, x : 1.6293

        model = Lorenz96I(n=n, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        sys = model.system(t, x)
        expected = forcing(t, x) * np.ones(n)
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([0.22, 0.288, 2.76, -2.1323])
        sys = model.system(t, x)
        expected = np.array([x[3] * (x[1] - x[2]) - x[0] + forcing(t, x),
                             x[0] * (x[2] - x[3]) - x[1] + forcing(t, x),
                             x[1] * (x[3] - x[0]) - x[2] + forcing(t, x),
                             x[2] * (x[0] - x[1]) - x[3] + forcing(t, x)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

    def test_system_definition_5d(self):
        n = 5
        forcing = lambda t, x : 9.35

        model = Lorenz96I(n=n, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        sys = model.system(t, x)
        expected = forcing(t, x) * np.ones(n)
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([4.537, -6.82, 2.11, -3.2, -12.23])
        sys = model.system(t, x)
        expected = np.array([x[4] * (x[1] - x[3]) - x[0] + forcing(t, x),
                             x[0] * (x[2] - x[4]) - x[1] + forcing(t, x),
                             x[1] * (x[3] - x[0]) - x[2] + forcing(t, x),
                             x[2] * (x[4] - x[1]) - x[3] + forcing(t, x),
                             x[3] * (x[0] - x[2]) - x[4] + forcing(t, x)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

class TestLorenz96II(unittest.TestCase):
    def test_system_definition_1x_1y(self):
        n = 1
        m = 1
        h = np.random.rand()
        c = 2 * np.random.rand()
        b = 1 + np.random.rand()
        forcing = lambda t, x, y : 2

        model = Lorenz96II(n=n, m=m, h=h, c=c, b=b, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        y = np.zeros((n, m))
        sys = model.system(t, np.concatenate([x, np.reshape(y, (n * m,))]))
        expected = np.concatenate(
            [forcing(t, x, y) * np.ones(n), np.zeros(n * m)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([4.6286])
        y = np.array([[-1.3879]])
        sys = model.system(t, np.concatenate([x, np.reshape(y, (n * m,))]))
        expected = np.concatenate(
            [-x + forcing(t, x, y) - h * c * np.sum(y[0,:]) / b,
             np.reshape(-c * y, (n * m,)) + h * c * x / b])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

    def test_system_definition_1x_2y(self):
        n = 1
        m = 2
        h = np.random.rand()
        c = -np.random.rand()
        b = 4 + np.random.rand()
        forcing = lambda t, x, y : -3.24

        model = Lorenz96II(n=n, m=m, h=h, c=c, b=b, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        y = np.zeros((n, m))
        sys = model.system(t, np.concatenate([x, np.reshape(y, (n * m,))]))
        expected = np.concatenate(
            [forcing(t, x, y) * np.ones(n), np.zeros(n * m)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([1.72])
        y = np.array([[-1.58, 1.41]])
        sys = model.system(t, np.concatenate([x, np.reshape(y, (n * m,))]))
        expected = np.concatenate(
            [-x + forcing(t, x, y) - h * c * np.sum(y[0,:]) / b,
             np.array([c * b * y[0,1] * (y[0,1] - y[0,0]) - c * y[0,0]
                       + h * c * x[0] / b,
                       c * b * y[0,0] * (y[0,0] - y[0,1]) - c * y[0,1]
                       + h * c * x[0] / b])])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

    def test_system_definition_1x_5y(self):
        n = 1
        m = 5
        h = np.random.rand()
        c = 15 * np.random.rand()
        b = -2 + np.random.rand()
        forcing = lambda t, x, y : 2

        model = Lorenz96II(n=n, m=m, h=h, c=c, b=b, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        y = np.zeros((n, m))
        sys = model.system(t, np.concatenate([x, np.reshape(y, (n * m,))]))
        expected = np.concatenate(
            [forcing(t, x, y) * np.ones(n), np.zeros(n * m)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([-1.1])
        y = np.array([[1.81, -1.42, -1.78,  4.72, 11.36]])
        sys = model.system(t, np.concatenate([x, np.reshape(y, (n * m,))]))
        expected = np.concatenate(
            [-x + forcing(t, x, y) - h * c * np.sum(y[0,:]) / b,
             np.array([c * b * y[0,1] * (y[0,4] - y[0,2]) - c * y[0,0]
                       + h * c * x[0] / b,
                       c * b * y[0,2] * (y[0,0] - y[0,3]) - c * y[0,1]
                       + h * c * x[0] / b,
                       c * b * y[0,3] * (y[0,1] - y[0,4]) - c * y[0,2]
                       + h * c * x[0] / b,
                       c * b * y[0,4] * (y[0,2] - y[0,0]) - c * y[0,3]
                       + h * c * x[0] / b,
                       c * b * y[0,0] * (y[0,3] - y[0,1]) - c * y[0,4]
                       + h * c * x[0] / b])])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

    def test_system_definition_3x_5y(self):
        n = 4
        m = 5
        h = np.random.rand()
        c =  np.random.rand()
        b = 5 + np.random.rand()
        forcing = lambda t, x, y : -4.3

        model = Lorenz96II(n=n, m=m, h=h, c=c, b=b, F=forcing)

        t = np.random.rand()

        x = np.zeros(n)
        y = np.zeros((n, m))
        sys = model.system(t, np.concatenate([x, np.reshape(y, (n * m,))]))
        expected = np.concatenate(
            [forcing(t, x, y) * np.ones(n), np.zeros(n * m)])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

        x = np.array([-2.76, -5.04,  1.82, 0.])
        y = np.array([[-2.07, -5.58, -2.39,  2.01,  6.02],
                      [-6.0, -0.25, -2.67, 7.63, 3.5],
                      [-3.13,  0.5, -7.12,  3.0,  9.67],
                      [-0.34, 6.95, -3.58, -1.20, -2.66]])
        sys = model.system(t, np.concatenate([x, np.reshape(y, (n * m,))]))
        expected = np.concatenate(
            [np.array([x[3] * (x[1] - x[2]) - x[0] + forcing(t, x, y)
                       - h * c * np.sum(y[0,:]) / b,
                       x[0] * (x[2] - x[3]) - x[1] + forcing(t, x, y)
                       - h * c * np.sum(y[1,:]) / b,
                       x[1] * (x[3] - x[0]) - x[2] + forcing(t, x, y)
                       - h * c * np.sum(y[2,:]) / b,
                       x[2] * (x[0] - x[1]) - x[3] + forcing(t, x, y)
                       - h * c * np.sum(y[3,:]) / b]),
             np.array([c * b * y[0,1] * (y[3,4] - y[0,2]) - c * y[0,0]
                       + h * c * x[0] / b,
                       c * b * y[0,2] * (y[0,0] - y[0,3]) - c * y[0,1]
                       + h * c * x[0] / b,
                       c * b * y[0,3] * (y[0,1] - y[0,4]) - c * y[0,2]
                       + h * c * x[0] / b,
                       c * b * y[0,4] * (y[0,2] - y[1,0]) - c * y[0,3]
                       + h * c * x[0] / b,
                       c * b * y[1,0] * (y[0,3] - y[1,1]) - c * y[0,4]
                       + h * c * x[0] / b,
                       c * b * y[1,1] * (y[0,4] - y[1,2]) - c * y[1,0]
                       + h * c * x[1] / b,
                       c * b * y[1,2] * (y[1,0] - y[1,3]) - c * y[1,1]
                       + h * c * x[1] / b,
                       c * b * y[1,3] * (y[1,1] - y[1,4]) - c * y[1,2]
                       + h * c * x[1] / b,
                       c * b * y[1,4] * (y[1,2] - y[2,0]) - c * y[1,3]
                       + h * c * x[1] / b,
                       c * b * y[2,0] * (y[1,3] - y[2,1]) - c * y[1,4]
                       + h * c * x[1] / b,
                       c * b * y[2,1] * (y[1,4] - y[2,2]) - c * y[2,0]
                       + h * c * x[2] / b,
                       c * b * y[2,2] * (y[2,0] - y[2,3]) - c * y[2,1]
                       + h * c * x[2] / b,
                       c * b * y[2,3] * (y[2,1] - y[2,4]) - c * y[2,2]
                       + h * c * x[2] / b,
                       c * b * y[2,4] * (y[2,2] - y[3,0]) - c * y[2,3]
                       + h * c * x[2] / b,
                       c * b * y[3,0] * (y[2,3] - y[3,1]) - c * y[2,4]
                       + h * c * x[2] / b,
                       c * b * y[3,1] * (y[2,4] - y[3,2]) - c * y[3,0]
                       + h * c * x[3] / b,
                       c * b * y[3,2] * (y[3,0] - y[3,3]) - c * y[3,1]
                       + h * c * x[3] / b,
                       c * b * y[3,3] * (y[3,1] - y[3,4]) - c * y[3,2]
                       + h * c * x[3] / b,
                       c * b * y[3,4] * (y[3,2] - y[0,0]) - c * y[3,3]
                       + h * c * x[3] / b,
                       c * b * y[0,0] * (y[3,3] - y[0,1]) - c * y[3,4]
                       + h * c * x[3] / b])])
        self.assertTrue(np.allclose(sys, expected, rtol=1.e-10, atol=1.e-10))

