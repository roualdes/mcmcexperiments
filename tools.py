import cmdstanpy as csp
import hashlib as hl
import numpy as np
import pandas as pd

import itertools
import logging

log = logging.getLogger(__name__)

def mean_sq_jumps(draws, digits = 3):
    """Return mean of squared distances between consecutive draws."""
    M = np.shape(draws)[0]
    jumps = draws[range(1, M), :] - draws[range(0, M - 1), :]
    return np.round(np.mean([np.dot(jump, jump) for jump in jumps]), digits)

def root_mean_square_error(theta, theta_sd, theta_hat, digits = 3):
    """Compute the standardized (z-score) RMSE for theta_hat given the reference mean and standard deviation."""
    return np.round(np.sqrt(np.sum(((theta_hat - theta) / theta_sd) ** 2) / np.size(theta)), digits)


def get_stan_files(cfg):
    "From a Hydra config file, initialize stan and data file paths"
    stan_file = f"{cfg["model_path"]}/{cfg["model_name"]}.stan"
    data_file = f"{cfg["model_path"]}/{cfg["model_name"]}.json"
    return stan_file, data_file

def stan_initializations(cfg):
    """Run Stan as the initializer for stepsize and theta.
    """

    stan_file, data_file = get_stan_files(cfg)

    csp_model = csp.CmdStanModel(stan_file = stan_file)
    fit = csp_model.sample(data = data_file,
                           metric = "unit_e",
                           adapt_delta = 0.95,
                           # parallel_chains = 1,
                           iter_warmup = cfg["init"]["warmup"],
                           iter_sampling = cfg["init"]["iterations"],
                           show_progress = False,
                           show_console = False)

    draws = fit.draws(concat_chains = True)[:, 7:]
    num_draws = np.shape(draws)[0]
    rng = np.random.default_rng()
    if cfg["replications"] > 0:
        draw_idx = rng.choice(num_draws, size = cfg["replications"])
    else:
        draw_idx = rng.choice(num_draws)

    m = np.mean(draws, axis = 0)
    s = np.std(draws, ddof = 1, axis = 0)

    draws2 = draws ** 2
    sq_m = np.mean(draws2, axis = 0)
    sq_s = np.std(draws2, axis = 0)

    inits = {
        "param_names": list(fit.column_names[7:]),
        "stepsize": fit.step_size[0], #  * 0.5,
        "initial_theta_constrained": np.copy(draws[draw_idx, :]),
        "mean": m,
        "std": s,
        "sq_mean": sq_m,
        "sq_std": sq_s,
    }

    print_fit(cfg, inits)

    return inits

def run_stan(cfg, gold_standard):
    "Run Stan for comparisons"

    stan_file, data_file = get_stan_files(cfg)

    csp_model = csp.CmdStanModel(stan_file = stan_file)
    fit = csp_model.sample(data = data_file,
                           chains = 1,
                           metric = "unit_e",
                           step_size = gold_standard["stepsize"],
                           iter_sampling = cfg["iterations"],
                           iter_warmup = cfg["warmup"],
                           adapt_engaged = False,
                           show_progress = False,
                           show_console = False)

    draws = fit.draws(concat_chains = True)
    leapfrog_steps = np.sum(draws[:, 4])

    draws = draws[:, 7:]        # skip non-parameter columns
    m = np.mean(draws, axis = 0)
    s = np.std(draws, ddof = 1, axis = 0)

    draws2 = draws ** 2
    sq_m = np.mean(draws2, axis = 0)
    sq_s = np.std(draws2, axis = 0)

    gs_m = gold_standard["mean"]
    gs_s = gold_standard["std"]
    gs_sq_m = gold_standard["sq_mean"]
    gs_sq_s = gold_standard["sq_std"]

    out_dict = {
        # "algorithm": "Stan",
        "model": cfg["model_name"],
        "mean": m,
        "std": s,
        "rmse": root_mean_square_error(gs_m, gs_s, m),
        "rmse_sq": root_mean_square_error(gs_sq_m, gs_sq_s, s),
        "msjd": mean_sq_jumps(draws),
        "steps": leapfrog_steps,
        "prop_accepted": 1.0,
    }

    print_fit(cfg, out_dict, gold_standard)

    return out_dict

def run_gist(cfg, gist_sampler, gold_standard):
    "Run GIST sampler for comparisons"

    M = cfg["iterations"]
    warmup = cfg["warmup"]
    samples = gist_sampler.sample_constrained(M + warmup)

    post_warmup_samples = samples[warmup:]
    m = np.mean(post_warmup_samples, axis = 0)
    s = np.std(post_warmup_samples, ddof = 1, axis = 0)

    gs_m = gold_standard["mean"]
    gs_s = gold_standard["std"]
    gs_sq_m = gold_standard["sq_mean"]
    gs_sq_s = gold_standard["sq_std"]

    out_dict = {
        # "algorithm": gist_sampler.sampler_name,
        # "model": cfg.model.name,
        "mean": m,
        "std": s,
        "rmse": root_mean_square_error(gs_m, gs_s, m),
        "rmse_sq": root_mean_square_error(gs_sq_m, gs_sq_s, s),
        "msjd": mean_sq_jumps(post_warmup_samples),
        "steps": gist_sampler.steps,
        "prop_accepted": gist_sampler.prop_accepted,
    }

    if hasattr(gist_sampler, "switch_prop"):
        out_dict["switch_prop"] = gist_sampler.switch_prop

    if hasattr(gist_sampler, "step_mean"):
        out_dict["step_mean"] = gist_sampler.step_mean

    print_fit(cfg, out_dict, gold_standard)

    return out_dict


def print_fit(cfg, output_dict, gold_standard = None):
    if cfg["print"]:
        print()
        # print(f"algorithm: {output_dict['algorithm']}")

        digits = cfg["digits"]

        if "prop_accepted" in output_dict:
            print(f"P(accepted) = {output_dict['prop_accepted']}")

        if "mean" in output_dict:
            # print(f"mean = {np.round(output_dict['mean'], digits)}")

            if gold_standard is not None:
                print(f"P(mean > m) = {np.round(np.mean(output_dict['mean'] > gold_standard['mean']), digits)}")

        if "std" in output_dict:
            print(f"std = {np.round(output_dict['std'], digits)}")

            if gold_standard is not None:
                print(f"P(std > s) = {np.round(np.mean(output_dict['std'] > gold_standard['std']), digits)}")

        if "rmse" in output_dict:
            print(f"rmse = {np.round(output_dict['rmse'], digits)}")

        if "rmse_sq" in output_dict:
            print(f"rmse (sq) = {np.round(output_dict['rmse_sq'], digits)}")

        if "msjd" in output_dict:
            print(f"msjd = {np.round(output_dict['msjd'], digits)}")

        if "steps" in output_dict:
            print(f"steps = {np.round(output_dict['steps'], digits)}")

        if "switch_prop" in output_dict:
            prop = output_dict["switch_prop"]
            total = np.sum(prop)
            print(f"switch_mean = {np.round(prop / total, digits)}")

        if "step_mean" in output_dict:
            sm = output_dict["step_mean"]
            print(f"step_mean = {np.round(sm, digits)}")

def md5hash(x):
    return hl.md5(x.encode("utf-8")).hexdigest()

def initialize_dataframe(algorithms, models, num_replications):
    """Initialize a DataFrame with zeros using lists of model names,
    algorithm names, and the number of replications."""

    reps = np.arange(1, num_replications + 1, dtype = np.int64).tolist()

    df = pd.DataFrame.from_records(itertools.product(algorithms, models, reps),
                                   columns = ["algorithm", "model", "replication"])

    ldf = len(df)
    df["rmse"] = np.zeros(ldf)
    df["rmse_sq"] = np.zeros(ldf)
    df["msjd"] = np.zeros(ldf)
    df["steps"] = np.zeros(ldf)
    df["prop_accepted"] = np.zeros(ldf)

    df.index = (df["algorithm"]
                .str.cat(df[["model", "replication"]].astype(str), sep = "-")
                .apply(lambda x: md5hash(x)))

    return df


def logsubexp(a, b):
    if a > b:
        return a + np.log1p(-np.exp(b - a))
    elif a < b:
        return b + np.log1p(-np.exp(a - b))
    else:
        return np.inf
