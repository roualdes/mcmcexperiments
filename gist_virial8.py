import numpy as np

import hmc
import traceback

class GISTV8(hmc.HMCBase):
    """Virial GIST

    Propose from foward trajectory and balance with backward
    trajectory.  switch_limit can be greater than 1. Proposals are
    biased towards the end of the trajectory in segments consisting of
    segment_length steps.  Biasing analogous to Stan.
    """
    def __init__(self, model, stepsize, theta = None, seed = None,
                 switch_limit = 1, segment_length = 2, **kwargs):

        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = f"GIST-V8_{switch_limit}"

        if theta is not None:
            self.theta = theta

        self.segment_length = segment_length
        self.switch_limit = switch_limit
        self.switches_passed = 0

        self.steps = 0
        self.proposal_steps = 0
        self.forward_steps = []
        self.mean_proposal_steps = 0.0

        self.backward_proportion = 0.0

        self.acceptance_probability = 0.0
        self.divergences = 0
        self.draws = 0

    def prepare_forward_pass(self):
        self.switches_passed = 0
        self.proposal_steps = 0

    def store_forward_steps(self):
        self.forward_steps.append(self.steps)

    def store_proposal_steps(self):
        d = self.proposal_steps - self.mean_proposal_steps
        self.mean_proposal_steps += d / self.draws

    def store_backward_proportion(self, b):
        d = b - self.backward_proportion
        self.backward_proportion += d / self.draws

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

    def trajectory(self, theta, rho, lsw = 0.0):
        theta_star = theta
        theta_prop = theta

        v = self.virial(theta, rho)
        sv = np.sign(v)
        H0 = self.log_joint(theta, rho)

        steps = 0
        steps_star = 0
        steps_prop = 0

        lsw_segment = -np.inf
        end_segment = False

        lsw_star = -np.inf
        lsw_prop = -np.inf

        switches = 0
        switches_star = 0
        switches_prop = 0

        while True:
            steps += 1
            end_segment = steps % self.segment_length == 0

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

            # multioulli: sample within segment
            lsw_segment = np.logaddexp(lsw_segment, delta)
            log_alpha = delta - lsw_segment
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_star = theta
                lsw_star = lsw
                steps_star = steps
                switches_star = switches

            # biased: sample segment
            if end_segment:
                log_beta = lsw_segment - lsw
                if np.log(self.rng.uniform()) < np.minimum(0.0, log_beta):
                    theta_prop = theta_star
                    lsw_prop = lsw_star
                    steps_prop = steps_star
                    switches_prop = switches_star
                lsw_segment = -np.inf

            # track total energy
            lsw = np.logaddexp(lsw, delta)

            if end_segment:
                if switches >= self.switch_limit - self.switches_passed:
                    break

        self.steps += steps
        self.switches_passed = switches_prop
        self.proposal_steps = steps_prop
        return theta_prop, lsw_prop, lsw

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)

            # forward pass
            self.prepare_forward_pass()
            theta_star, lsw_star, FW = self.trajectory(theta, rho)

            # only for comparisons, otherwise unnecessary
            self.store_forward_steps()
            self.store_proposal_steps()

            # backward pass not always necessary
            if self.switches_passed < self.switch_limit:
                _, _, BW  = self.trajectory(theta, -rho, lsw = lsw_star)
                BW = self.logsubexp(BW, 0.0) # don't double count theta0
                self.store_backward_proportion(1)
            else:
                self.store_backward_proportion(0)
                BW = FW

            # H_star - H_0 + (H_0 - BW) - (H_star - FW)
            log_alpha = FW - BW

            accepted = 0
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                accepted = 1
                self.theta = theta_star

            # only for comparisons, otherwise unnecessary
            self.store_acceptance_probability(accepted)
        except Exception as e:
            traceback.print_exc()
            pass
        return self.theta
