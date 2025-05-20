import duckdb

from pathlib import Path

import cmdstanpy as csp
import numpy as np
import pandas as pd
import tools as tls

csp_path = Path.home() / "cmdstan"
csp.set_cmdstan_path(str(csp_path))
tls.stop_griping()

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


db = duckdb.connect("goldstandard.db")

for model in models:
    print(f"MODEL: {model}")

    print("building model...", end = "\r")
    sf, df = tls.get_stan_files(model)
    stan_model = csp.CmdStanModel(stan_file = sf, force_compile = True)

    fit = stan_model.sample(data = df,
                            adapt_delta = 0.9,
                            chains = 1,
                            iter_warmup = 25_000,
                            iter_sampling = 50_000,
                            show_progress = False,
                            show_console = False)

    draws_df = fit.draws_pd()
    db.sql(f"CREATE TABLE '{model}' AS SELECT * FROM draws_df")

db.close()
