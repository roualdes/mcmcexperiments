import duckdb

from pathlib import Path

import cmdstanpy as csp
import numpy as np
import pandas as pd
import tools as tls

csp_path = Path.home() / "cmdstan"
csp.set_cmdstan_path(str(csp_path))
tls.stop_griping()

# TODO
# set seeds
# increase warmup iterations, so as to reduce issues

models = [
    "normal",
    "ill-normal",
    "corr-normal",
    # "rosenbrock",
    "glmm-poisson",
    "hmm",
    "garch",
    # "lotka-volterra",
    # "arma",
    "arK"
]

replications = 20
db = tls.init_db("results.db")
algo = "bias++"

for model in models:
    print(f"MODEL: {model}")

    print("building model...", end = "\r")
    sf, df = tls.get_stan_files(model)
    stan_model = csp.CmdStanModel(stan_file = sf, force_compile = True)

    for rep in range(replications):
        print(f"REP: {rep}\033[K", end = "\r")

        fit = stan_model.sample(data = df,
                                iter_warmup = 10_000,
                                iter_sampling = 5_000,
                                show_progress = False,
                                show_console = False)

        fs = fit.summary()
        draws_df = fit.draws_pd()
        draws = draws_df.values[:, 10:]
        leapfrogs = np.sum(draws_df["n_leapfrog__"])
        min_stepsize = np.min(draws_df["stepsize__"])

        stats_df = pd.DataFrame({
            "algorithm": [algo],
            "model": [model],
            "rep": [rep],
            "min_ess_bulk": [np.min(fs["ESS_bulk"])],
            "min_ess_tail": [np.min(fs["ESS_tail"])],
            "max_rhat": [np.max(fs["R_hat"])],
            "msjd": [tls.mean_sq_jumps(draws)],
            "leapfrog": [int(leapfrogs)],
            "min_stepsize": [min_stepsize]
        })

        db.sql("INSERT INTO stats BY NAME SELECT * FROM stats_df")

db.close()
