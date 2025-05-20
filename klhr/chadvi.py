from bsmodel import BSModel
from pathlib import Path
from scipy.optimize import minimize

import bridgestan as bs
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import json
import yaml

import tools as tls

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)


bs.set_bridgestan_path(Path(cfg["bs_path"]).expanduser())
# stan_file, data_file = tls.get_stan_files(cfg)

bs_model = BSModel(stan_file = "stan/one_t.stan",
                   data_file = "stan/one_t.json")

def logq(z, m, s):
    d = z - m
    return -0.5 * np.log(2 * np.pi * s * s) - 0.5 * np.sum(d * d) / (s * s)

# just can't seem to get this to work well
# but I don't understand why not
class CHADVI():
    def __init__(self, bsmodel, seed = None):
        self.bsmodel = bsmodel
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def _f(self, theta, z):
        w = np.zeros_like(z)
        mw = -np.inf
        for n, zn in enumerate(z):
            x = theta[0] + np.exp(theta[1]) * zn
            w[n] = self.bsmodel.log_density(x) - logq(x, theta[0], np.exp(theta[1]))
            if w[n] > mw: mw = w[n]
        return np.mean(np.exp(w) ** 2 - 1)

    def fit(self, N = 20, m = 0, s = 1):
        init = np.array([m, s])
        z = self.rng.normal(size = (N, 1))
        o = minimize(self._f, init, args = (z,))
        return np.array([o.x[0], np.exp(o.x[1])])


chadvi = CHADVI(bs_model)
m, s = chadvi.fit(N = 100)

xx = np.linspace(90, 120, 301)
fxx = [np.exp(bs_model.log_density(np.array([xxn]), propto=False)) for xxn in xx]
plt.clf()
plt.plot(xx, fxx, label = "model")
plt.plot(xx, st.norm(loc=m, scale=s).pdf(xx), label = "approx")
plt.legend()
plt.savefig("dadvi.png")
plt.close()
