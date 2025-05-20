from bsmodel import BSModel
from pathlib import Path
from randomwalk import RW
from gist_uturn2 import GISTU2
from gist_virial4 import GISTV4
from gist_virial8 import GISTV8
from gist_virial10 import GISTV10

import bridgestan as bs
import cmdstanpy as csp
import matplotlib.pyplot as plt
import numpy as np
import tools as tls

import json
import pprint
import yaml
import time
import sys


with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

pprint.pp(cfg)
bs.set_bridgestan_path(Path(cfg["bs_path"]).expanduser())
stan_file, data_file = tls.get_stan_files(cfg)

bs_model = BSModel(stan_file = stan_file, data_file = data_file)

################ GISTV4 ################

print("GISTV4")
algo = GISTV4(bs_model, cfg["stepsize"], cfg["seed"],
              switch_limit = cfg["switch_limit"])

t0 = time.time()
d = algo.sample_constrained(cfg["warmup"] + cfg["iterations"])
dt = time.time() - t0

thetas = d["thetas"][cfg["warmup"]:, :]
m = np.mean(thetas, axis = 0)
s = np.var(thetas, ddof=1, axis = 0)
msjd = tls.mean_sq_jumps(thetas)

print(f"mean = {np.round(m, cfg['digits'])}")
print(f"std = {np.round(s, cfg['digits'])}")
print(f"msjd = {msjd}")
print(f"time = {dt}")
print(f"steps = {d['steps']}")
print(f"acceptance rate = {d['acceptance_rate']}")
print(f"mean_proposal_steps = {d['mean_proposal_steps']}")

plt.hist(d["forward_steps"],
         color = "green", label = algo.sampler_name,
         histtype = "step", density = True)

################ GISTV10 ################

print("GISTV10")
algo = GISTV10(bs_model, cfg["stepsize"], cfg["seed"],
              switch_limit = cfg["switch_limit"],
              segment_length = cfg["segment_length"])

t0 = time.time()
d = algo.sample_constrained(cfg["warmup"] + cfg["iterations"])
dt = time.time() - t0

thetas = d["thetas"][cfg["warmup"]:, :]
m = np.mean(thetas, axis = 0)
s = np.var(thetas, ddof=1, axis = 0)
msjd = tls.mean_sq_jumps(thetas)

print(f"mean = {np.round(m, cfg['digits'])}")
print(f"std = {np.round(s, cfg['digits'])}")
print(f"msjd = {msjd}")
print(f"time = {dt}")
print(f"steps = {d['steps']}")
print(f"acceptance rate = {np.round(d['acceptance_rate'], cfg['digits'])}")
print(f"mean_proposal_steps = {np.round(d['mean_proposal_steps'], cfg['digits'])}")

plt.hist(d["forward_steps"],
         color = "blue", label = algo.sampler_name,
         histtype = "step", density = True)

################ GISTU ################

print("\nGISTU2")
algo = GISTU2(bs_model, cfg["stepsize"], cfg["seed"])

t0 = time.time()
d = algo.sample_constrained(cfg["warmup"] + cfg["iterations"])
dt = time.time() - t0

thetas = d["thetas"][cfg["warmup"]:, :]
m = np.mean(thetas, axis = 0)
s = np.var(thetas, ddof=1, axis = 0)
msjd = tls.mean_sq_jumps(thetas)

print(f"mean = {np.round(m, cfg['digits'])}")
print(f"std = {np.round(s, cfg['digits'])}")
print(f"msjd = {msjd}")
print(f"time = {dt}")
print(f"steps = {d['steps']}")
print(f"acceptance rate = {d['acceptance_rate']}")
print(f"mean_proposal_steps = {d['mean_proposal_steps']}")

plt.hist(d["forward_steps"],
         label = "GIST-U2", color = "orange",
         histtype = "step", density = True)

plt.xlabel("forward_steps")
plt.title(f"cfg['model_name']")
plt.legend()
plt.savefig(f"forward_steps_{cfg['model_name']}.png")
