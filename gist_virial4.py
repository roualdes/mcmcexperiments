import numpy as np

import hmc
import traceback

class GISTV4(hmc.HMCBase):
    """Virial GIST

    Propose from foward trajectory and balance with backward
    trajectory.  switch_limit can be greater than 1. Online
    Multinoulli sampling during foward trajectory.

    """
    def __init__(self, model, stepsize,
                 theta = None, seed = None,
                 switch_limit = 1, **kwargs):

        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = f"GIST-V4_{switch_limit}"

        if theta is not None:
            self.theta = theta

        self.switch_limit = switch_limit
        self.switches_passed = 0

        self.steps = 0
        self.proposal_steps = 0
        self.forward_steps = []
        self.mean_proposal_steps = 0.0

        self.acceptance_probability = 0.0
        self.divergences = 0
        self.draws = 0

    def prepare_forward_pass(self):
        self.switches_passed = 0
        self.proposal_steps = 0

    def store_forward_steps(self, steps):
        self.forward_steps.append(steps)

    def store_proposal_steps(self, steps):
        d = steps - self.mean_proposal_steps
        self.mean_proposal_steps += d / self.draws

    def store_acceptance_probability(self, accepted):
        d = accepted - self.acceptance_probability
        self.acceptance_probability += d / self.draws

    def virial(self, theta, rho):
        return 2 * rho.dot(theta)

    def logsubexp(self, a, b):
        if a > b:
            return a + np.log1p(-np.exp(b - a))
        elif a < b:
            return b + np.log1p(-np.exp(a - b))
        else:
            return np.inf

    def trajectory(self, theta, rho, lsw = 0.0,
                   forward = True):
        theta_prop = theta

        v = self.virial(theta, rho)
        sv = np.sign(v)
        H0 = self.log_joint(theta, rho)

        steps = 0
        steps_prop = 0

        lsw_prop = -np.inf

        switches = 0
        switches_prop = 0

        while True:
            steps += 1

            theta, rho = self.leapfrog_step(theta, rho)
            H = self.log_joint(theta, rho)
            delta = H - H0
            if np.abs(-delta) > 50.0:
                self.divergences += 1
                break

            # check virial sign switch
            v = self.virial(theta, rho)
            if sv * np.sign(v) < 0:
                switches += 1
                sv = np.sign(v)

            # track total energy
            lsw = np.logaddexp(lsw, delta)

            # multinoulli: sample within trajectory
            log_alpha = delta - lsw
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_prop = theta
                lsw_prop = lsw
                steps_prop = steps
                switches_prop = switches

            if switches >= self.switch_limit - self.switches_passed:
                break

        self.steps += steps
        if forward:
            self.switches_passed = switches_prop
            self.store_forward_steps(steps)
        return theta_prop, lsw_prop, steps_prop, lsw

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)

            # forward pass
            self.prepare_forward_pass()
            theta_star, lsw_star, steps_star, FW = self.trajectory(theta, rho, forward = True)

            # backward pass
            _, _, _, BW = self.trajectory(theta, -rho, lsw = lsw_star)
            BW = self.logsubexp(BW, 0.0) # don't double count theta0

            # H_star - H_0 + (H_0 - BW) - (H_star - FW)
            log_alpha = FW - BW

            accepted = 0
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                accepted = 1
                self.theta = theta_star
                self.store_proposal_steps(steps_star)

            # only for comparisons, otherwise unnecessary
            self.store_acceptance_probability(accepted)
        except Exception as e:
            traceback.print_exc()
            pass
        return self.theta
