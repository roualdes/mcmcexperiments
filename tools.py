import cmdstanpy as csp
import hashlib as hl
import numpy as np
import pandas as pd

import duckdb
import itertools
import logging
import warnings

def stop_griping():
    warnings.filterwarnings(
        "ignore", message="Loading a shared object .* that has already been loaded.*"
    )
    csp.utils.get_logger().setLevel(logging.ERROR)

def mean_sq_jumps(draws, digits = 4):
    """
    Return mean of squared distances between consecutive draws.
    """

    M = np.shape(draws)[0]
    jumps = draws[range(1, M), :] - draws[range(0, M - 1), :]
    return np.round(np.mean([np.dot(jump, jump) for jump in jumps]), digits)

def root_mean_square_error(theta_hat, theta, theta_sd, digits = 4):
    """
    Compute the standardized (z-score) RMSE for theta_hat given the
    reference mean and standard deviation.
    """

    z = (theta_hat - theta) / theta_sd
    rmse = np.sqrt(np.mean(z * z))
    return np.round(rmse, digits)

def get_stan_files(cfg):
    "From config file, initialize stan and data file paths"
    stan_file = f"{cfg['model_path']}/{cfg['model_name']}.stan"
    data_file = f"{cfg['model_path']}/{cfg['model_name']}.json"
    return stan_file, data_file

def stan_initializations(cfg):
    """
    Run Stan as the initializer for stepsize and theta.
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
    draws2 = draws ** 2

    num_draws = np.shape(draws)[0]
    rng = np.random.default_rng()
    draw_idx = rng.choice(num_draws, size = cfg["replications"])

    cfg["param_names"] = list(fit.column_names[7:])
    cfg["stepsize"] = fit.step_size[0] * 0.5
    cfg["initial_theta_constrained"] = np.copy(draws[draw_idx, :])
    cfg["gs_m"] = np.mean(draws, axis = 0)
    cfg["gs_s"] = np.std(draws, ddof = 1, axis = 0)
    cfg["gs_sq_m"] = np.mean(draws2, axis = 0)
    cfg["gs_sq_s"] = np.std(draws2, axis = 0)

def run_stan(cfg):
    """Run Stan for comparisons"""

    stan_file, data_file = get_stan_files(cfg)

    csp_model = csp.CmdStanModel(stan_file = stan_file)
    fit = csp_model.sample(data = data_file,
                           chains = 1,
                           metric = "unit_e",
                           step_size = cfg["stepsize"],
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
    m2 = np.mean(draws ** 2, axis = 0)

    out_dict = {
        "algorithm": "Stan",
        # "mean": m,
        # "sd": s,
        "rmse": root_mean_square_error(m, cfg["gs_m"], cfg["gs_s"]),
        "rmse_sq": root_mean_square_error(m2, cfg["gs_sq_m"], cfg["gs_sq_s"]),
        "msjd": mean_sq_jumps(draws),
        "steps": leapfrog_steps,
        "stepsize": cfg["stepsize"],
        "mean_proposal_steps": -1,
        "acceptance_rate": 1.0,
        "switch_limit": -1,
    }

    return out_dict

def run_gist(cfg, gist_sampler):
    "Run GIST sampler for comparisons"

    M = cfg["iterations"]
    warmup = cfg["warmup"]
    d = gist_sampler.sample_constrained(M + warmup)

    samples = d["thetas"]
    post_warmup_samples = samples[warmup+1:]

    m = np.mean(post_warmup_samples, axis = 0)
    s = np.std(post_warmup_samples, ddof = 1, axis = 0)
    m2 = np.mean(post_warmup_samples ** 2, axis = 0)

    out_dict = {
        "algorithm": gist_sampler.sampler_name,
        # "mean": m,
        # "sd": s,
        "rmse": root_mean_square_error(m, cfg["gs_m"], cfg["gs_s"]),
        "rmse_sq": root_mean_square_error(m2, cfg["gs_sq_m"], cfg["gs_sq_s"]),
        "msjd": mean_sq_jumps(post_warmup_samples),
        "steps": d["steps"],
        "stepsize": cfg["stepsize"],
        "mean_proposal_steps": d["mean_proposal_steps"],
        "acceptance_rate": d["acceptance_rate"],
        "switch_limit": gist_sampler.switch_limit if hasattr(gist_sampler, "switch_limit") else -1
    }

    return out_dict

def store_run(db, cfg, out):
    out["rep"] = cfg["rep"]
    out["model"] = cfg["model_name"]
    o = {k : [v] for k, v in out.items()}
    df = pd.DataFrame(o)
    db.sql("INSERT INTO stats BY NAME SELECT * FROM df")

def init_db(cfg):
    db = duckdb.connect(cfg["db"])
    db.sql("""CREATE TABLE IF NOT EXISTS stats (
        algorithm VARCHAR,
        model VARCHAR,
        rep INTEGER,
        rmse DOUBLE,
        rmse_sq DOUBLE,
        msjd DOUBLE,
        steps BIGINT,
        stepsize DOUBLE,
        mean_proposal_steps DOUBLE,
        acceptance_rate DOUBLE,
        switch_limit INTEGER
        )""")
    return db
