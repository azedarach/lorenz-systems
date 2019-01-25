import numpy as np
from scipy.integrate import solve_ivp

class Lorenz63(object):
    def __init__(self, sigma=10.0, rho=28.0, beta=2.667):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def system(self, t, x):
        if np.size(x) != 3:
            raise ValueError("incorrect state size")

        return np.array([self.sigma * (x[1] - x[0]),
                         self.rho * x[0] - x[1] - x[0] * x[2],
                         x[0] * x[1] - self.beta * x[2]])

    def evolve(self, x0, t1, t0=0, **kwargs):
        if np.size(x) != 3:
            raise ValueError("incorrect initial state size")

        func = lambda t, x : self.system(t, x)
        sol = solve_ivp(func, (t0, t1), x0, **kwargs)
        if not sol.success:
            raise RuntimeError(sol.reason)
        return (sol.t, sol.y)
