from pathlib import Path
import sys

import bridgestan as bs
import click
import cmdstanpy as csp
import numpy as np
import pandas as pd

import summary as smry

@click.command()
@click.option("-M", "--iterations", "iterations", type=int, default=1_000)
@click.option("-w", "--warmup", "warmup", type=int, default=100)
@click.option("-v", "--verbose", "verbose", is_flag=True)
@click.option("-f", "--stepsizefactor", type=float, default=1.0)
@click.argument("model", type=str)
def main(iterations, warmup, stepsizefactor, verbose, model):

    stan_dir = Path(__file__).resolve().parent / "stan"
    models = [f.stem for f in stan_dir.glob("*.stan") if f.is_file()]
    if model not in models:
        print(f"Unknown model: {model}")
        print(f"Available models are: {models}")
        sys.exit(0)

    csp.set_cmdstan_path(str(Path.home() / "cmdstan"))
    msg = "Loading a shared object .* that has already been loaded.*"
    warnings.filterwarnings("ignore", message=msg)
    csp.utils.get_logger().setLevel(logging.ERROR)

    stan_file = stan_dir / f"{model}.stan"
    data_file = stan_dir / f"{model}.json"
    stan_model = csp.CmdStanModel(stan_file = stan_file)

    with open(stan_dir / f"{model}.yaml", "r") as tf:
        truth = yaml.safe_load(tf)

    init_constrained_theta = truth["init_constrained_theta"][1]
    param_names = truth["parameter_names"]
    init_theta = {pn: truth["init_constrained_theta"][n]
                  for n, pn in enumerate(param_names)}

    csp_model = csp.CmdStanModel(stan_file = stan_file)
    fit = csp_model.sample(data = data_file,
                           chains = 1,
                           # seed = seed,
                           metric = "unit_e",
                           step_size = stepsizefactor * truth["stepsize"],
                           inits = init_theta,
                           iter_sampling = iterations,
                           iter_warmup = warmup,
                           adapt_engaged = False,
                           show_progress = False,
                           show_console = False)

    draws = fit.draws(concat_chains = True)
    leapfrog_steps = np.sum(draws[:, 4])
    samples = draws[:, 7:]        # skip non-parameter columns

    m = np.mean(samples, axis = 0)
    s = np.var(samples, ddof = 1, axis = 0)
    m2 = np.mean(samples ** 2, axis = 0)

    rmse = smry.rmse(m, truth["mean"], truth["std"])
    rmse_sq = smry.rmse(m2, truth["sq_mean"], truth["sq_std"])
    msjd = smry.msjd(samples)
    essx = smry.ess(samples[:, :, np.newaxis])
    essx2 = smry.ess(samples[:, :, np.newaxis])

    steps = leapfrog_steps
    acceptance_rate = 1.0

    if verbose:
        digits = 2
        # print(f"mean: {np.round(m, digits)}")
        # print(f"std: {np.round(s, digits)}")

        # print(f"rmse: {np.round(rmse, digits)}")
        # print(f"msjd: {np.round(msjd, digits)}")
        # print(f"essx: {np.round(essx, digits)}")
        # print(f"essx2: {np.round(essx2, digits)}")

        # print(f"steps: {steps}")
        print(f"acceptance rate: {acceptance_rate}")


if __name__ == "__main__":
    main()
