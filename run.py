from bsmodel import BSModel
from pathlib import Path
from randomwalk import RW

import bridgestan as bs
import cmdstanpy as csp
import matplotlib.pyplot as plt
import numpy as np
import tools as tls

import json
import pprint
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

pprint.pp(cfg)
bs.set_bridgestan_path(Path(cfg["bs_path"]).expanduser())
stan_file, data_file = tls.get_stan_files(cfg)

bs_model = BSModel(stan_file = stan_file,
                   data_file = data_file)

rw = RW(bs_model, cfg["stepsize"], cfg["seed"])
thetas = rw.sample_constrained(cfg["iterations"])

thetas = thetas[cfg["warmup"]:, :]
m = thetas.mean()
s = thetas.std()

print(f"mean = {np.round(m, cfg['digits'])}")
print(f"std = {np.round(s, cfg['digits'])}")
