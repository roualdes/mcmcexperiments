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

import click

import bridgestan as bs
import cmdstanpy as csp
import numpy as np
import tools as tls
import pandas as pd
import summary as smry

config = {
    "iterations": 1_000,
    "warmup": 1_000,
    "model_path": "../stan",
    "verbose": False,
    "stepsize_factor": 1,
    "output": "./output/algorithm_by_model",
    "seed": 204,
    "replications": 20,
}

@click.command()
@click.argument("algorithm", type=str)
@click.argument("model", type=str)
def main(algorithm, model):

    algorithms = {
        "stan": "",
        "gistum": GISTUM,
        "gistvb": GISTVB,
        "gistvd": GISTVD,
        "gistvm": GISTVM
    }
    if algorithm not in algorithms.keys():
        print(f"unknown algorithm: {algorithm}")
        sys.exit(0)

    if config["verbose"]:
        print(f"ALGORITHM: {algorithm}\033[K")
    alg = algorithms[algorithm]

    models = [f.stem for f in Path(config["model_path"]).glob("*.stan") if f.is_file()]
    if model not in models:
        print(f"Unknown model: {model}")
        print(f"Available models are: {models}")
        sys.exit(0)

    config["model_name"] = model
    if config["verbose"]:
        print(f"MODEL: {model}\033[K")

    bs.set_bridgestan_path(Path().home() / "bridgestan")
    csp.set_cmdstan_path(str(Path().home() / "cmdstan"))
    if config["verbose"]:
        print("building model...\033[K", end = "\r")

    stan_file = f"{config['model_path']}/{model}.stan"
    data_file = f"{config['model_path']}/{model}.json"
    bs_model = BSModel(stan_file = stan_file, data_file = data_file)

    tls.stop_griping()
    tls.goldstandard_details(config)
    results = tls.create_results_container()

    for rep in range(config["replications"]):
        if config["verbose"]:
            print(f"rep: {rep}\033[K", end = "\r")

        config["seed"] += rep

        init_constrained_theta = config["init_constrained_theta"][rep]

        if algorithm == "stan":
            config["init_theta"] = init_constrained_theta
            out = tls.run_stan(config)
        else:
            init_unc_theta = bs_model.unconstrain(np.array(init_constrained_theta))
            g = alg(bs_model,
                    config["stepsize_factor"] * config["stepsize"],
                    theta = init_unc_theta,
                    seed = config["seed"])
            out = tls.run_gist(config, g)

        out["rep"] = rep
        tls.update_all_results(results, out)
        pd.DataFrame(results).to_parquet(f"{config['output']}/{algorithm}_{model}.parquet")

if __name__ == "__main__":
    main()
