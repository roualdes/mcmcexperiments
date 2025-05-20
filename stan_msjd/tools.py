import duckdb
import logging
import warnings

import numpy as np
import cmdstanpy as csp


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

def get_stan_files(model_name):
    "From config file, initialize stan and data file paths"
    stan_file = f"../stan/{model_name}.stan"
    data_file = f"../stan/{model_name}.json"
    return stan_file, data_file

def init_db(db_path):
    db = duckdb.connect(db_path)
    db.sql("""CREATE TABLE IF NOT EXISTS stats (
        algorithm VARCHAR,
        model VARCHAR,
        rep INTEGER,
        msjd DOUBLE,
        leapfrog BIGINT,
        min_stepsize DOUBLE,
        min_ess_bulk DOUBLE,
        min_ess_tail DOUBLE,
        max_rhat DOUBLE,
        )""")
    return db

def init_gsdb(db_path, models):


    return db
