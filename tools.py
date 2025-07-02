import summary as smry

import cmdstanpy as csp
import numpy as np
import pandas as pd

import logging
import warnings
import yaml

def stop_griping():
    warnings.filterwarnings(
        "ignore", message="Loading a shared object .* that has already been loaded.*"
    )
    csp.utils.get_logger().setLevel(logging.ERROR)

def goldstandard_details(cfg):
    experiment_file = f"{cfg['model_path']}/{cfg['model_name']}.yaml"
    with open(experiment_file, "r") as infile:
        gs = yaml.safe_load(infile)
    cfg.update(gs)

def run_stan(cfg):
    """Run Stan for comparisons"""

    stan_file = f"{cfg['model_path']}/{cfg['model_name']}.stan"
    data_file = f"{cfg['model_path']}/{cfg['model_name']}.json"

    param_names = cfg["param_names"]
    init_theta = {pn: cfg["init_theta"][n] for n, pn in enumerate(param_names)}

    csp_model = csp.CmdStanModel(stan_file = stan_file)
    fit = csp_model.sample(data = data_file,
                           chains = 1,
                           seed = cfg["seed"],
                           metric = "unit_e",
                           step_size = cfg["stepsize_factor"] * cfg["stepsize"],
                           inits = init_theta,
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
        "mean": m,
        "std": s,
        "rmse": smry.root_mean_square_error(m, cfg["gs_m"], cfg["gs_s"]).tolist(),
        "rmse_sq": smry.root_mean_square_error(m2, cfg["gs_sq_m"], cfg["gs_sq_s"]).tolist(),
        "msjd": smry.mean_sq_jumps(draws).tolist(),
        "essx": smry.ess_basic(draws[:, :, np.newaxis]).tolist(),
        "essx2": smry.ess_basic(draws[:, :, np.newaxis] ** 2).tolist(),
        "steps": leapfrog_steps,
        "stepsize": cfg["stepsize"],
        "forward_steps": [-1],
        "mean_stop_steps": [-1],
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
        "algorithm": d["sampler_name"],
        "mean": m,
        "std": s,
        "rmse": smry.root_mean_square_error(m, cfg["gs_m"], cfg["gs_s"]).tolist(),
        "rmse_sq": smry.root_mean_square_error(m2, cfg["gs_sq_m"], cfg["gs_sq_s"]).tolist(),
        "msjd": smry.mean_sq_jumps(post_warmup_samples).tolist(),
        "essx": smry.ess_basic(post_warmup_samples[:, :, np.newaxis]).tolist(),
        "essx2": smry.ess_basic(post_warmup_samples[:, :, np.newaxis] ** 2).tolist(),
        "steps": d["steps"],
        "forward_steps": d["forward_steps"],
        "mean_stop_steps": d["mean_stopping_steps"],
        "mean_proposal_steps": d["mean_proposal_steps"],
        "stepsize": cfg["stepsize"],
        "acceptance_rate": d["acceptance_rate"],
        "switch_limit": d["switch_limit"] if hasattr(d, "switch_limit") else -1
    }

    return out_dict

def create_results_container():
    slots = ["algorithm", "mean", "std",
             "rmse", "rmse_sq", "msjd",
             "essx", "essx2", "steps",
             "forward_steps", "mean_stop_steps",
             "mean_proposal_steps", "stepsize",
             "acceptance_rate", "switch_limit", "rep"]
    return {slot: [] for slot in slots}

def update_result(ky, outdict, results):
    if ky in outdict and ky in results:
        if isinstance(results[ky], type(np.zeros(1))):
            outdict[ky].append(results[ky].tolist())
        else:
            outdict[ky].append(results[ky])

def update_all_results(outdict, results):
    for ky in outdict.keys():
        update_result(ky, outdict, results)
