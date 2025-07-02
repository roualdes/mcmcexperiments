import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pathlib import Path

import click
import yaml

import cmdstanpy as csp
import numpy as np
import pandas as pd
import tools as tls


config = {
    "iterations": 20_000,
    "warmup": 50_000,
    "model_path": "../stan",
    "verbose": False,
    "stepsize_factor": 1,
    "output": "./output/algorithm_by_model",
    "seed": 95926,
    "replications": 20,
}

@click.command()
@click.argument("model", type=str)
def main(model):

    models = [f.stem for f in Path(config["model_path"]).glob("*.stan") if f.is_file()]
    if model not in models:
        print(f"Unknown model: {model}")
        print(f"Available models are: {models}")
        sys.exit(0)

    config["model_name"] = model
    if config["verbose"]:
        print(f"MODEL: {model}\033[K")

    csp.set_cmdstan_path(str(Path.home() / "cmdstan"))
    tls.stop_griping()

    if config["verbose"]:
        print("building model...\033[K", end = "\r")

    stan_file = f"{config['model_path']}/{model}.stan"
    data_file = f"{config['model_path']}/{model}.json"
    stan_model = csp.CmdStanModel(stan_file = stan_file, force_compile = True)

    if config["verbose"]:
        print("running model...\033[K", end = "\r")

    fit = stan_model.sample(data = data_file,
                                adapt_delta = 0.9,
                                seed = config["seed"],
                                chains = 1,
                                metric="unit_e",
                                iter_warmup = config["iterations"],
                                iter_sampling = config["warmup"],
                                show_progress = False,
                                show_console = False)

    draws_df = fit.draws_pd()
    draws = draws_df.iloc[:, 10:]

    experiment_data = {}

    # starting points
    rng = np.random.default_rng(config["seed"])
    idx = rng.integers(draws.shape[0], size = config["replications"])
    experiment_data["init_constrained_theta"] = draws.iloc[idx].values.copy().tolist()

    # stepsize
    experiment_data["stepsize"] = draws_df["stepsize__"].values[-1].tolist()

    # summary stats
    experiment_data["gs_m"] = np.mean(draws, axis = 0).tolist()
    experiment_data["gs_s"] = np.std(draws, ddof = 1, axis = 0).tolist()

    draws2 = draws ** 2
    experiment_data["gs_sq_m"] = np.mean(draws2, axis = 0).tolist()
    experiment_data["gs_sq_s"] = np.std(draws2, axis = 0).tolist()
    experiment_data["param_names"] = draws.columns.tolist()

    experiment_file = f"{config['model_path']}/{config['model_name']}.yaml"
    with open(experiment_file, "w") as outfile:
        yaml.dump(experiment_data, outfile)

if __name__ == "__main__":
    main()
