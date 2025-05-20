import cmdstanpy as csp
import numpy as np
import pandas as pd

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

# def stan_initializations(cfg):
#     """
#     Run Stan as the initializer for stepsize and theta.
#     """

#     stan_file, data_file = get_stan_files(cfg)

#     csp_model = csp.CmdStanModel(stan_file = stan_file)
#     fit = csp_model.sample(data = data_file,
#                            metric = "unit_e",
#                            adapt_delta = 0.95,
#                            # parallel_chains = 1,
#                            iter_warmup = cfg["init"]["warmup"],
#                            iter_sampling = cfg["init"]["iterations"],
#                            show_progress = False,
#                            show_console = False)

#     draws = fit.draws(concat_chains = True)[:, 7:]
#     draws2 = draws ** 2

#     num_draws = np.shape(draws)[0]
#     rng = np.random.default_rng()
#     draw_idx = rng.choice(num_draws, size = cfg["replications"])

#     cfg["param_names"] = list(fit.column_names[7:])
#     cfg["stepsize"] = fit.step_size[0] * 0.5
#     cfg["initial_theta_constrained"] = np.copy(draws[draw_idx, :])
#     cfg["gs_m"] = np.mean(draws, axis = 0)
#     cfg["gs_s"] = np.std(draws, ddof = 1, axis = 0)
#     cfg["gs_sq_m"] = np.mean(draws2, axis = 0)
#     cfg["gs_sq_s"] = np.std(draws2, axis = 0)

def goldstandard_details(cfg):
    df = pd.read_parquet(f"./output/goldstandard_{cfg['model_name']}.parquet")
    draws = df.iloc[:, 10:]

    rng = np.random.default_rng(cfg["seed"])
    idx = draws.shape[0]
    init_unc_theta = draws.iloc[rng.integers(idx, size = cfg["replications"])].values.copy()
    cfg["init_unc_theta"] = init_unc_theta

    cfg["stepsize"] = df["stepsize__"].values[-1]

    cfg["gs_m"] = np.mean(draws, axis = 0)
    cfg["gs_s"] = np.std(draws, ddof = 1, axis = 0)

    draws2 = draws ** 2
    cfg["gs_sq_m"] = np.mean(draws2, axis = 0)
    cfg["gs_sq_s"] = np.std(draws2, axis = 0)



def run_stan(cfg):
    """Run Stan for comparisons"""

    stan_file, data_file = get_stan_files(cfg)

    csp_model = csp.CmdStanModel(stan_file = stan_file)
    fit = csp_model.sample(data = data_file,
                           chains = 1,
                           adapt_engaged = False,
                           show_progress = False,
                           show_console = False)
    param_names = fit.draws_pd().columns[10:].to_list()
    init_theta = {pn: cfg["init_theta"][n] for n, pn in enumerate(param_names)}

    fit = csp_model.sample(data = data_file,
                           chains = 1,
                           metric = "unit_e",
                           step_size = cfg["stepsize"],
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
        "rmse": root_mean_square_error(m, cfg["gs_m"], cfg["gs_s"]),
        "rmse_sq": root_mean_square_error(m2, cfg["gs_sq_m"], cfg["gs_sq_s"]),
        "msjd": mean_sq_jumps(draws),
        "essx": ess_basic(draws[:, :, np.newaxis]),
        "essx2": ess_basic(draws[:, :, np.newaxis] ** 2),
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
        "algorithm": d["sampler_name"],
        "mean": m,
        "std": s,
        "rmse": root_mean_square_error(m, cfg["gs_m"], cfg["gs_s"]),
        "rmse_sq": root_mean_square_error(m2, cfg["gs_sq_m"], cfg["gs_sq_s"]),
        "msjd": mean_sq_jumps(post_warmup_samples),
        "essx": ess_basic(post_warmup_samples[:, :, np.newaxis]),
        "essx2": ess_basic(post_warmup_samples[:, :, np.newaxis] ** 2),
        "steps": d["steps"],
        "forward_steps": d["forward_steps"],
        "mean_stop_steps": d["mean_stopping_steps"],
        "mean_proposal_steps": d["mean_proposal_steps"],
        "stepsize": cfg["stepsize"],
        "acceptance_rate": d["acceptance_rate"],
        "switch_limit": d["switch_limit"] if hasattr(d, "switch_limit") else -1
    }

    return out_dict

def fft_nextgoodsize(N):
    if N <= 2:
        return 2

    while True:
        m = N

        while np.remainder(m, 2) == 0:
            m /= 2

        while np.remainder(m, 3) == 0:
            m /= 3

        while np.remainder(m, 5) == 0:
            m /= 5

        if m <= 1:
            return N

        N += 1

def autocovariance(x):
    N = np.size(x)
    Mt2 = 2 * fft_nextgoodsize(N)
    yc = np.zeros(Mt2)
    yc[:N] = x - np.mean(x)
    t = np.fft.fft(yc)
    ac = np.fft.fft(np.conj(t) * t)
    return np.real(ac)[:N] / (N * N * 2)


def isconstant(x):
    mn = np.min(x)
    mx = np.max(x)
    return np.isclose(mn, mx)


def _ess(x):
    niterations, nchains = np.shape(x)

    if niterations < 3:
        return np.nan

    if np.any(np.isnan(x)):
        return np.nan

    if np.any(np.isinf(x)):
        return np.nan

    if isconstant(x):
        return np.nan

    acov = np.apply_along_axis(autocovariance, 0, x)
    chain_mean = np.mean(x, axis = 0)
    mean_var = np.mean(acov[0, :]) * niterations / (niterations - 1)
    var_plus = mean_var * (niterations - 1) / niterations

    if nchains > 1:
        var_plus += np.var(chain_mean, ddof = 1)

    rhohat = np.zeros(niterations)
    rhohat_even = 1
    rhohat[0] = rhohat_even
    rhohat_odd = 1 - (mean_var - np.mean(acov[1, :])) / var_plus
    rhohat[1] = rhohat_odd

    t = 1
    while t < niterations - 4 and rhohat_even + rhohat_odd > 0:
        rhohat_even = 1 - (mean_var - np.mean(acov[t + 1, :])) / var_plus
        rhohat_odd = 1 - (mean_var - np.mean(acov[t + 2, :])) / var_plus

        if rhohat_even + rhohat_odd >= 0:
            rhohat[t + 1] = rhohat_even
            rhohat[t + 2] = rhohat_odd

        t += 2

    max_t = t
    if rhohat_even > 0:
        rhohat[max_t + 1] = rhohat_even

    t = 1
    while t <= max_t - 3:
        if rhohat[t + 1] + rhohat[t + 2] > rhohat[t - 1] + rhohat[t]:
            rhohat[t + 1] = (rhohat[t - 1] + rhohat[t]) / 2
            rhohat[t + 2] = rhohat[t + 1]
        t += 2

    ess = nchains * niterations
    tau = -1 + 2 * np.sum(rhohat[0:np.maximum(1, max_t)]) + rhohat[max_t + 1]
    tau = np.maximum(tau, 1 / np.log10(ess))
    return ess / tau

def ess_basic(x):
    N, D, C = np.shape(x)
    esses = np.zeros(D)
    for d in range(D):
        esses[d] = _ess(x[:, d, :])
    return esses
