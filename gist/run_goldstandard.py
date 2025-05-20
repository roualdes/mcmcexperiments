import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pathlib import Path

import cmdstanpy as csp
import numpy as np
import pandas as pd
import tools as tls

import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

csp_path = Path.home() / "cmdstan"
csp.set_cmdstan_path(str(csp_path))
tls.stop_griping()

if len(sys.argv) < 2:
    print("Please provide a string as a command-line argument.")
    sys.exit(0)


model = sys.argv[1]

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

print(f"MODEL: {model}\033[K", end = "\r")

cfg["model_name"] = model

print("building model...\033[K", end = "\r")
sf, df = tls.get_stan_files(cfg)
stan_model = csp.CmdStanModel(stan_file = sf, force_compile = True)

print("running model...\033[K", end = "\r")
fit = stan_model.sample(data = df,
                        adapt_delta = 0.9,
                        chains = 1,
                        iter_warmup = 25_000,
                        iter_sampling = 50_000,
                        show_progress = False,
                        show_console = False)

draws_df = fit.draws_pd()
draws_df.to_parquet(f"./output/goldstandard_{model}.parquet")
