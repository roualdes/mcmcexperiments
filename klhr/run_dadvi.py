from bsmodel import BSModel
from dadvi import DADVI
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

bs_model = BSModel(stan_file = "stan/one_exponential.stan",
                   data_file = "stan/one_exponential.json")

dadvi = DADVI(bs_model)
m, s = dadvi.fit()
theta

xx = np.linspace(-10, 10, 301)
fxx = [np.exp(bs_model.log_density(np.array([xxn]), propto=False)) for xxn in xx]
plt.clf()
plt.plot(xx, fxx, label = "model")
plt.plot(xx, st.norm(loc=m, scale=s).pdf(xx), label = "approx")
plt.plot(xx, st.norm(loc=m, scale=o["hess_inv"][0,0]).pdf(xx), label = "approx-lr")
# plt.hist(np.log(fit.draws_pd()["y"].values), density = True, histtype = "step")
plt.legend()
plt.savefig("dadvi.png")
plt.close()


# import cmdstanpy as csp

# model = csp.CmdStanModel(stan_file = "stan/one_exponential.stan")
# fit = model.sample(data = "stan/one_exponential.json", iter_sampling = 10_000)
# fit.summary()
