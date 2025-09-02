import logging
from pathlib import Path
import sys
import warnings

import click
import cmdstanpy as csp
import numpy as np
import pandas as pd
import tools as tls
import yaml



@click.command()
@click.option("-M", "--iterations", "iterations", type=int, default=2_000)
@click.option("-w", "--warmup", "warmup", type=int, default=1_000)
@click.option("-v", "--verbose", "verbose", is_flag=True)
@click.option("-s", "--seed", "seed", type=int, default=95929)
@click.option("-r", "--replications", "replications", type=int, default=20)
@click.option("-d", "--adaptdelta", "adaptdelta", type=float, default=0.9)
@click.argument("model", type=str)
def main(iterations, warmup, verbose, seed, replications, adaptdelta, model):

    stan_dir = Path(__file__).resolve().parent / "stan"
    models = [f.stem for f in stan_dir.glob("*.stan") if f.is_file()]
    if model not in models:
        print(f"Unknown model: {model}")
        print(f"Available models are: {models}")
        sys.exit(0)

    if verbose:
        print(f"model: {model}\033[K")

    csp.set_cmdstan_path(str(Path.home() / "cmdstan"))
    msg = "Loading a shared object .* that has already been loaded.*"
    warnings.filterwarnings("ignore", message=msg)
    csp.utils.get_logger().setLevel(logging.ERROR)

    if verbose:
        print("building model...\033[K", end = "\r")

    stan_file = stan_dir / f"{model}.stan"
    data_file = stan_dir / f"{model}.json"
    stan_model = csp.CmdStanModel(stan_file = stan_file,
                                  force_compile = True)

    if verbose:
        print("running model...\033[K", end = "\r")

    fit = stan_model.sample(data = data_file,
                            adapt_delta = adaptdelta,
                            seed = seed,
                            chains = 1,
                            metric="unit_e",
                            # save_warmup=True,
                            iter_warmup = iterations,
                            iter_sampling = warmup,
                            show_progress = False,
                            show_console = False)

    draws_df = fit.draws_pd()
    draws = draws_df.iloc[:, 10:]

    experiment_data = {}

    # starting points
    rng = np.random.default_rng(seed)
    idx = rng.integers(draws.shape[0], size = replications)
    experiment_data["init_constrained_theta"] = draws.iloc[idx].values.copy().tolist()

    # stepsize
    experiment_data["stepsize"] = draws_df["stepsize__"].values[-1].tolist()

    # summary stats
    experiment_data["mean"] = np.mean(draws, axis = 0).tolist()
    experiment_data["std"] = np.std(draws, ddof = 1, axis = 0).tolist()

    draws2 = draws ** 2
    experiment_data["sq_mean"] = np.mean(draws2, axis = 0).tolist()
    experiment_data["sq_std"] = np.std(draws2, axis = 0).tolist()
    experiment_data["parameter_names"] = draws.columns.tolist()

    with open(stan_dir / f"{model}.yaml", "w") as tf:
        yaml.dump(experiment_data, tf)

if __name__ == "__main__":
    main()
