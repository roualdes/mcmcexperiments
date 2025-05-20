import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from mcmc import MCMCBase
from bsmodel import BSModel
from dadvi import DADVI

from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss

import numpy as np

class KLHR(MCMCBase):
    def __init__(self, bsmodel, N = 20, theta = None, seed = None):
        super().__init__(bsmodel, -1, seed = seed)

        self.N = N
        self.x, self.w = hermgauss(self.N)

        self.m = np.zeros(self.D)
        self.S = np.eye(self.D)

        self.acceptance_probability = 0
        self._draw = 0

    def _LVI(self, theta, rho):
        out = 0.0
        for xn, wn in zip(self.x, self.w):
            zn = np.sqrt(2) * np.exp(theta[1]) * xn + theta[0]
            thetan = self._to_rho(zn, rho, 0) # self.theta)
            out += wn * self.model.log_density(thetan)
        return -out / np.sqrt(np.pi) - theta[1]

    def fit_dadvi(self, rho):
        o = minimize(self._LVI, np.zeros(2), args = (rho,))
        mkl = o.x[0]
        skl = np.sqrt(o["hess_inv"][0, 0])
        return mkl, skl

    def _random_direction(self):
        rho = self.rng.multivariate_normal(self.m, self.S)
        return rho / np.linalg.norm(rho)

    def _to_rho(self, x, rho, origin):
        return x.reshape(-1, 1) * rho + origin

    def _from_rho(self, x, rho):
        return (x.flatten() / rho)[0]

    def _log_normal(self, x, m, s):
        z = (x - m) / s
        return -np.log(s) - 0.5 * z * z

    def _metropolis_step(self, m, s, rho):
        mu = m - self._from_rho(self.theta, rho)
        zp = self.rng.normal(loc = mu, scale = s, size = 1)
        thetap = self._to_rho(zp, rho, self.theta)

        a = self.model.log_density(thetap) - self.model.log_density(self.theta)
        a += self._log_normal(self._from_rho(self.theta, rho), m, s)
        a -= self._log_normal(self._from_rho(thetap, rho), m, s)

        accept = np.log(self.rng.uniform()) < np.minimum(0, a)
        if accept:
            self.theta = thetap

        d = accept - self.acceptance_probability
        self.acceptance_probability += d / (self._draw)

    def draw(self):
        self._draw += 1
        rho = self._random_direction()
        mkl, skl = self.fit_dadvi(rho)
        self._metropolis_step(mkl, skl, rho)
        return self.theta
