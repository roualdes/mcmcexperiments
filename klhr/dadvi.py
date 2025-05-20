import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from bsmodel import BSModel
from pathlib import Path
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss

import bridgestan as bs
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import json
import yaml

import tools as tls


class DADVI():
    def __init__(self, bsmodel, seed = None):
        self.bsmodel = bsmodel
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.x, self.w = hermgauss(30)

    def _f(self, theta, z, rho, origin):
        out = 0.0
        for n, zn in enumerate(z):
            x = theta[0] + np.exp(theta[1]) * zn
            x = rotate_shift(x, rho, origin)
            out += (self.bsmodel.log_density(x) - out) / (n + 1)
        return -out - theta[1]

    def fit(self, z, rho, origin, m = 0, s = 1, init_scale = 0.1):
        o = minimize(self._f, np.array([m, s]),
                     args = (z * init_scale, rho, origin))
        mkl = o.x[0]
        # skl = np.exp(o.x[1]) * init_scale
        skl = o["hess_inv"][0, 0]
        return np.array([mkl, skl])
