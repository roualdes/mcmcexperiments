from pathlib import Path
import cmdstanpy as csp
import numpy as np
import pandas as pd
import tools as tls

csp_path = Path.home() / "cmdstan"
csp.set_cmdstan_path(str(csp_path))
tls.stop_griping()


models = [
    # "normal",
    # "ill-normal",
    # "corr-normal",
    # "rosenbrock",
    # "glmm-poisson",
    # "hmm",
    "garch",
    # "lotka-volterra",
    # "arma",
    # "arK"
]


sf, df = tls.get_stan_files("garch")
stan_model = csp.CmdStanModel(stan_file = sf, force_compile = True)


initd = {"mu": 5.2448769, "alpha0": 1.471980, "alpha1": 0.567725, "beta1": 0.291822}
inits = [initd for _ in range(4)]
M = {"inv_metric": [0.0148331, 0.155055, 0.30102, 1.87743]}

fit = stan_model.sample(data = df,
                        # adapt_delta = 0.7 if algo == "bias++" else 0.8,
                        iter_warmup = 10_000,
                        iter_sampling = 5_000,
                        seed = 739,
                        inits = inits,
                        metric = M,
                        step_size = 0.35,
                        adapt_engaged = False,
                        show_progress = False,
                        show_console = False)

fs = fit.summary()
draws_df = fit.draws_pd()
draws = draws_df.values[:, 10:]
leapfrogs = np.sum(draws_df["n_leapfrog__"])
min_stepsize = np.mean(draws_df["stepsize__"])
