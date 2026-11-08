import numpy as np
from scipy.optimize import minimize
from plant import plant as twin_plant

class MPCController:
    def __init__(self, a, c, horizon=5, Q=1.0, R=0.1, u_max=1.0):
        self.a = a; self.c = c; self.horizon = horizon
        self.Q = Q; self.R = R; self.u_max = u_max; self.n = 8
    def reset(self): pass
    def act(self, x, t, rate):
        u0 = np.zeros(self.horizon * self.n)
        x_bar = twin_plant.x_bar
        def cost(u_seq):
            u_seq = u_seq.reshape(self.horizon, self.n)
            cv = 0.0; x_pred = x.copy()
            for k in range(self.horizon):
                x_pred = x_bar + self.a * (x_pred - x_bar) + rate - self.c * u_seq[k]
                err = x_pred - x_bar
                cv += self.Q * np.sum(err**2) + self.R * np.sum(u_seq[k]**2)
            return cv
        bounds = [(-self.u_max, self.u_max)] * (self.horizon * self.n)
        res = minimize(cost, u0, bounds=bounds, method='L-BFGS-B', options={'maxiter': 10})
        return res.x[:self.n]
