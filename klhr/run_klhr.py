import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from bsmodel import BSModel
from dadvi import DADVI
from klhr import KLHR

from pathlib import Path
import bridgestan as bs
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import tools as tls
import json
import yaml


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
bs.set_bridgestan_path(Path(cfg["bs_path"]).expanduser())
# stan_file, data_file = tls.get_stan_files(cfg)

bs_model = BSModel(stan_file = "../stan/normal.stan",
                   data_file = "../stan/normal.json")

M = 10_000

algo = KLHR(bs_model)
draws = algo.sample(M)

print(f"mean = {np.mean(draws, axis = 0)}")
print(f"std = {np.std(draws, axis = 0, ddof = 1)}")


xx = np.linspace(-10, 10, 301)
fxx = [np.exp(bs_model.log_density(np.array([xxn]), propto=False)) for xxn in xx]
plt.clf()
plt.hist(draws[:, 0], histtype= "step", density = True, label = "klhr")
plt.plot(xx, fxx, label = "model")
plt.legend()
plt.savefig("klhr_histogram.png")
plt.close()

plt.clf()
idx = np.arange(M)
plt.plot(idx, draws)
plt.savefig("klhr_traceplot.png")
plt.close()
