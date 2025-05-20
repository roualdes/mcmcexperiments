import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from bsmodel import BSModel
from pathlib import Path
from gist_uturn_multinoulli import GISTUM
from gist_virial_biased import GISTVB
from gist_virial_distance import GISTVD
from gist_virial_multinoulli import GISTVM

import bridgestan as bs
import cmdstanpy as csp
import numpy as np
import tools as tls
import pandas as pd

import json
import pprint
import yaml
import time

if len(sys.argv) < 3:
    print("Please provide a model and an algorithm as arguments.")
    sys.exit(0)

algo = sys.argv[1]
model = sys.argv[2]

algos = {
    "stan": "",
    "gistum": GISTUM,
    "gistvb": GISTVB,
    "gistvd": GISTVD,
    "gistvm": GISTVM
}

algorithm = algos[algo]

if algo not in algos.keys():
    print("Unknown algorithm: {algo}.")
    sys.exit(0)

print(f"ALGO: {algo}\033[K")

models = [
    "normal",
    "ill-normal",
    "corr-normal",
    "rosenbrock",
    "glmm-poisson",
    "hmm",
    "garch",
    "lotka-volterra",
    "arma",
    "arK"
]

if model not in models:
    print("Unknown model: {model}.")
    sys.exit(0)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
# pprint.pp(cfg)

tls.stop_griping()
bs.set_bridgestan_path(Path(cfg["bs_path"]).expanduser())

print(f"MODEL: {model}\033[K")
cfg["model_name"] = model

print("building model...\033[K", end = "\r")
stan_file, data_file = tls.get_stan_files(cfg)
bs_model = BSModel(stan_file = stan_file, data_file = data_file)

tls.goldstandard_details(cfg)

if algo == "stan":

    for rep in range(cfg["replications"]):
        print(f"rep: {rep}\033[K", end = "\r")
        init_theta = bs_model.unconstrain(cfg["init_unc_theta"][rep])
        cfg["init_theta"] = init_theta
        sout = tls.run_stan(cfg)
        pd.DataFrame([sout]).to_parquet(f"./output/{algo}_{model}.parquet")

else:

    for rep in range(cfg["replications"]):
        print(f"rep: {rep}\033[K", end = "\r")
        init_theta = bs_model.unconstrain(cfg["init_unc_theta"][rep])
        g = algorithm(bs_model, cfg["stepsize"], theta = init_theta)
        gout = tls.run_gist(cfg, g)
        pd.DataFrame([gout]).to_parquet(f"./output/{algo}_{model}.parquet")
