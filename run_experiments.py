from bsmodel import BSModel
from pathlib import Path
from randomwalk import RW
from gist_uturn2 import GISTU2
from gist_uturn3 import GISTU3
from gist_virial4 import GISTV4
from gist_virial5 import GISTV5

import bridgestan as bs
import cmdstanpy as csp
import matplotlib.pyplot as plt
import numpy as np
import tools as tls

import duckdb
import json
import pprint
import yaml
import time

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# pprint.pp(cfg)
bs.set_bridgestan_path(Path(cfg["bs_path"]).expanduser())

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

db = tls.init_db(cfg)
tls.stop_griping()

for model in models:
    print(f"MODEL NAME: {model}")
    cfg["model_name"] = model

    print("building model...", end = "\r")
    stan_file, data_file = tls.get_stan_files(cfg)
    bs_model = BSModel(stan_file = stan_file, data_file = data_file)

    print("Stan initialization...", end = "\r")
    tls.stan_initializations(cfg)

    for rep in range(cfg["replications"]):
        print(f"REP: {rep}\033[K", end = "\r")
        cfg["rep"] = rep
        cfg["init_theta"] = bs_model.unconstrain(cfg["initial_theta_constrained"][rep])

        out_stan = tls.run_stan(cfg)
        gistv5 = GISTV5(bs_model, cfg["stepsize"],
                        theta = cfg["init_theta"], switch_limit = 2)
        out_gistv5_2 = tls.run_gist(cfg, gistv5)
        tls.store_run(db, cfg, out_gistv5_2)

        gistv5_3 = GISTV5(bs_model, cfg["stepsize"],
                          theta = cfg["init_theta"], switch_limit = 3)
        out_gistv5_3 = tls.run_gist(cfg, gistv5_3)
        tls.store_run(db, cfg, out_gistv5_3)

        gistu = GISTU2(bs_model, cfg["stepsize"],
                      theta = cfg["init_theta"])
        out_gistu = tls.run_gist(cfg, gistu)
        tls.store_run(db, cfg, out_gistu)

db.close()
