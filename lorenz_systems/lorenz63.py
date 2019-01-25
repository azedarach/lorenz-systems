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

class CoupledLorenz63(object):
    def __init__(self, sigma=10.0, rho=28.0, beta=2.667,
                 ce=0.08, c=1.0, cz=1.0, k1=10.0, k2=-11.0,
                 tau=0.1, s=1.0):

        if np.isscalar(sigma):
            self.sigma = sigma * np.ones(3)
        else:
            self.sigma = sigma

        if np.isscalar(rho):
            self.rho = rho * np.ones(3)
        else:
            self.rho = rho

        if np.isscalar(beta):
            self.beta = beta * np.ones(3)
        else:
            self.beta = beta

        self.ce = ce
        self.c = c
        self.cz = cz
        self.k1 = k1
        self.k2 = k2
        self.tau = tau
        self.s = s

    def system(self, t, x):
        if np.size(x) != 9:
            raise ValueError("incorrect state size")

        dxdt = np.zeros(9)
        dxdt[0] = (self.sigma[0] * (x[1] - x[0]) - self.ce *
                   (self.s * x[3] + self.k1))
        dxdt[1] = (self.rho[0] * x[0] - x[1] - x[0] * x[2]
                   + self.ce * (self.s * x[4] + self.k1))
        dxdt[2] = (x[0] * x[1] - self.beta[0] * x[2])
        dxdt[3] = (self.sigma[1] * (x[4] - x[3]) - self.c *
                   (self.s * x[6] + self.k2) - self.ce *
                   (self.s * x[0] + self.k1))
        dxdt[4] = (self.rho[1] * x[3] - x[4] - x[3] * x[5]
                   + self.c * (self.s * x[7] + self.k2)
                   + self.ce * (self.s * x[1] + self.k1))
        dxdt[5] = (x[3] * x[4] - self.beta[1] * x[5]
                   + self.cz * x[8])
        dxdt[6] = (self.tau * self.sigma[2] * (x[7] - x[6])
                   - self.c * (x[3] + self.k2))
        dxdt[7] = (self.tau * self.rho[2] * x[6] - self.tau * x[7]
                   - self.tau * self.s * x[6] * x[8]
                   + self.c * (x[4] + self.k2))
        dxdt[8] = (self.tau * self.s * x[6] * x[7] - self.tau
                   * self.beta[2] * x[8] - self.cz * x[5])

        return dxdt

    def evolve(self, x0, t1, t0=0, **kwargs):
        if np.size(x) != 9:
            raise ValueError("incorrect initial state size")

        func = lambda t, x : self.system(t, x)
        sol = solve_ivp(func, (t0, t1), x0, **kwargs)
        if not sol.success:
            raise RuntimeError(sol.reason)
        return (sol.t, sol.y)
