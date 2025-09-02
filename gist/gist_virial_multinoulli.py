import numpy as np

import hmc
import traceback

class GISTVM(hmc.HMCBase):
    """GIST Virial

    Sample along trajectory using multinoulli

    """
    def __init__(self, model, stepsize,
                 theta = None, seed = None,
                 switch_limit = 3, **kwargs):

        super().__init__(model, stepsize, theta = theta, seed = seed)
        self.sampler_name = "GIST-Virial-Multinoulli"

        self.switch_limit = switch_limit
        self.switches_passed = 0

        # only for comparisons, otherwise unnecessary
        self.steps = 0
        self.forward_steps = []
        self.mean_proposal_steps = 0.0
        self.mean_stopping_steps = 0.0
        self.acceptance_rate = 0.0
        self.divergences = 0
        self.draws = 0

    def update_forward_steps(self, steps):
        self.forward_steps.append(steps)

    def update_proposal_steps(self, steps):
        d = steps - self.mean_proposal_steps
        self.mean_proposal_steps += d / self.draws

    def update_stopping_steps(self, steps):
        d = steps - self.mean_stopping_steps
        self.mean_stopping_steps += d / self.draws

    def update_acceptance_rate(self, accepted):
        d = accepted - self.acceptance_rate
        self.acceptance_rate += d / self.draws

    def prepare_forward_pass(self):
        self.switches_passed = 0

    def virial(self, theta, rho):
        return 2 * rho.dot(theta)

    def logsubexp(self, a, b):
        if a > b:
            return a + np.log1p(-np.exp(b - a))
        elif a < b:
            return b + np.log1p(-np.exp(a - b))
        else:
            return np.inf

    def trajectory(self, theta, rho, lsw = 0.0, forward = False):
        theta_prop = theta
        rho_prop = rho

        v = self.virial(theta, rho)
        sv = np.sign(v)
        H0 = self.log_joint(theta, rho)

        steps_stop = 0
        steps_prop = 0

        lsw_prop = -np.inf

        switches = 0
        switches_prop = 0

        while True:
            steps_stop += 1

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
                rho_prop = rho
                lsw_prop = lsw
                steps_prop = steps_stop
                switches_prop = 0 # switches

            if switches >= self.switch_limit - self.switches_passed:
                break

        self.steps += steps_stop
        if forward:
            self.switches_passed = switches_prop
            self.update_forward_steps(steps_stop)
            self.update_stopping_steps(steps_stop)
            self.update_proposal_steps(steps_prop)

        return theta_prop, lsw_prop, steps_prop, steps_stop, lsw

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)

            self.prepare_forward_pass()
            theta_star, lsw_star, F, _, FW = self.trajectory(theta, rho, forward = True)
            _, _, _, B, BW = self.trajectory(theta, -rho, lsw = lsw_star)
            BW = self.logsubexp(BW, 0.0) # don't double count theta0

            if not(1 <= F and 0 <= B):
                self.update_acceptance_rate(False)
                return self.theta

            # H_star - H_0 + (H_0 - BW) - (H_star - FW)
            log_alpha = FW - BW

            accepted = np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha)
            if accepted:
                self.theta = theta_star

            self.update_acceptance_rate(accepted)
        except Exception as e:
            traceback.print_exc()
            pass
        return self.theta
