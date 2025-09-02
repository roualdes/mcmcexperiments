from pathlib import Path
import sys

import bridgestan as bs
import click
import cmdstanpy as csp
import numpy as np
import pandas as pd
import yaml

from bsmodel import BSModel
from gist_uturn_multinoulli import GISTUM
from gist_virial_biased import GISTVB
from gist_virial_distance import GISTVD
from gist_virial_multinoulli import GISTVM
import tools as tls
import summary as smry

@click.command()
@click.option("-M", "--iterations", "iterations", type=int, default=1_000)
@click.option("-w", "--warmup", "warmup", type=int, default=100)
@click.option("-v", "--verbose", "verbose", is_flag=True)
@click.option("-f", "--stepsizefactor", type=float, default=1.0)
@click.argument("algorithm", type=str)
@click.argument("model", type=str)
def main(iterations, warmup, stepsizefactor, verbose, algorithm, model):

    algorithms = {"gistvb": GISTVB, "gistvm": GISTVM, "gistum": GISTUM}
    if algorithm not in algorithms.keys():
        print(f"Unknown algorithm: {algorithm}")
        print(f"Available algorithms are: {list(algorithms.keys())}")
        sys.exit(0)

    stan_dir = Path(__file__).resolve().parent / "stan"
    models = [f.stem for f in stan_dir.glob("*.stan") if f.is_file()]
    if model not in models:
        print(f"Unknown model: {model}")
        print(f"Available models are: {models}")
        sys.exit(0)

    bs.set_bridgestan_path(Path().home().expanduser() / "bridgestan")

    stan_file = stan_dir / f"{model}.stan"
    data_file = stan_dir / f"{model}.json"
    bs_model = BSModel(stan_file = stan_file,
                       data_file = data_file)

    with open(stan_dir / f"{model}.yaml", "r") as tf:
        truth = yaml.safe_load(tf)

    init_constrained_theta = truth["init_constrained_theta"][1]
    init_theta = bs_model.unconstrain(np.array(init_constrained_theta))
    gist = algorithms[algorithm](bs_model,
                                 stepsizefactor * truth["stepsize"],
                                 theta = init_theta)

    gist_output = gist.sample_constrained(iterations + warmup)
    samples = gist_output["thetas"]
    post_warmup_samples = samples[warmup+1:]

    m = np.mean(post_warmup_samples, axis = 0)
    s = np.var(post_warmup_samples, ddof = 1, axis = 0)
    m2 = np.mean(post_warmup_samples ** 2, axis = 0)

    rmse = smry.rmse(m, truth["mean"], truth["std"])
    rmse_sq = smry.rmse(m2, truth["sq_mean"], truth["sq_std"])
    msjd = smry.msjd(post_warmup_samples)
    essx = smry.ess(post_warmup_samples[:, :, np.newaxis])
    essx2 = smry.ess(post_warmup_samples[:, :, np.newaxis])

    steps = gist_output["steps"]
    mean_proposal_steps = gist_output["mean_proposal_steps"]
    mean_stopping_steps = gist_output["mean_stopping_steps"]
    acceptance_rate = gist_output["acceptance_rate"]


    if verbose:
        digits = 4
        print(f"mean: {np.round(m, digits)}")
        print(f"std: {np.round(s, digits)}")

        # print(f"rmse: {np.round(rmse, digits)}")
        # print(f"msjd: {np.round(msjd, digits)}")
        # print(f"essx: {np.round(essx, digits)}")
        # print(f"essx2: {np.round(essx2, digits)}")

        # print(f"steps: {steps}")
        # print(f"mean proposal steps: {mean_proposal_steps}")
        # print(f"mean stopping steps: {mean_stopping_steps}")
        print(f"acceptance rate: {acceptance_rate}")


if __name__ == "__main__":
    main()
