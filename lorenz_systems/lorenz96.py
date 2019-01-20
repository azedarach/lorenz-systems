import numpy as np
from scipy.integrate import solve_ivp

class Lorenz96I(object):
    def __init__(self, n=8, F=(lambda t, x : 0)):
        if n < 1:
            raise ValueError("n must be an integer greater than or equal to 1")

        self.n = int(n)
        self.F = F

    def system(self, t, x):
        if np.size(x) != self.n:
            raise ValueError("incorrect state size")

        xim1 = np.roll(x, 1)
        xip1 = np.roll(x, -1)
        xim2 = np.roll(x, 2)
        return xim1 * (xip1 - xim2) - x + self.F(t, x)

    def evolve(self, x0, t1, t0=0, **kwargs):
        if np.size(x0) != self.n:
            raise ValueError("incorrect initial state size")

        func = lambda t, x : self.system(t, x)
        sol = solve_ivp(func, (t0, t1), x0, **kwargs)
        if not sol.success:
            raise RuntimeError(sol.reason)
        return (sol.t, sol.y)

class Lorenz96II(object):
    def __init__(self, n=8, m=4, h=1, b=10, c=10, F=(lambda t, x, y : 0)):
        if n < 1:
            raise ValueError("n must be an integer greater than or equal to 1")
        if m < 1:
            raise ValueError("m must be an integer greater than or equal to 1")

        self.n = int(n)
        self.m = int(m)
        self.h = h
        self.b = b
        self.c = c
        self.F = F

    def system(self, t, x):
        if np.size(x) != self.n * (self.m + 1):
            raise ValueError("incorrect state size")

        h = self.h
        b = self.b
        c = self.c

        xt = x[0:self.n]
        yt = x[self.n:]
        yt_mat = np.reshape(x[self.n:], (self.n, self.m))

        xtim1 = np.roll(xt, 1)
        xtip1 = np.roll(xt, -1)
        xtim2 = np.roll(xt, 2)
        dxtdt = (xtim1 * (xtip1 - xtim2) - xt + self.F(t, xt, yt_mat)
                 - h * c * np.sum(yt_mat, axis=1) / b)

        ytjp1 = np.roll(yt, -1)
        ytjm1 = np.roll(yt, 1)
        ytjp2 = np.roll(yt, -2)
        dytdt = (c * b * ytjp1 * (ytjm1 - ytjp2) - c * yt
                 + h * c * np.repeat(xt, self.m) / b)

        return np.concatenate([dxtdt, dytdt])

    def evolve(self, x0, y0, t1, t0=0, **kwargs):
        if np.size(x0) != self.n:
            raise ValueError("incorrect size for initial values x0")
        if np.size(y0) != self.n * self.m:
            raise ValueError("incorrect size for initial values y0")

        if y0.shape == (self.m * self.n,):
            X0 = np.concatenate([x0, y0])
        elif y0.shape == (self.m * self.n, 1):
            X0 = np.concatenate([x0, np.ravel(y0)])
        elif y0.shape == (self.n, self.m):
            X0 = np.concatenate(
                [x0, np.reshape(y0, self.m * self.n)])

        func = lambda t, x : self.system(t, x)
        sol = solve_ivp(func, (t0, t1), X0, **kwargs)
        if not sol.success:
            raise RuntimeError(sol.reason)

        xt_sol = sol.y[:self.n,:]
        yt_sol = np.reshape(sol.y[self.n:,:], (self.n, self.m, xt_sol.shape[1]))

        return (sol.t, xt_sol, yt_sol)

