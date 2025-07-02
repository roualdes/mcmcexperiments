import numpy as np

import hmc
import traceback

class GISTVB(hmc.HMCBase):
    """GIST Virial

    Sample along trajectory using biased progressive sampling with
    segments of size segment_length

    """
    def __init__(self, model, stepsize, theta = None, seed = None,
                 switch_limit = 3, segment_length = 2, **kwargs):

        super().__init__(model, stepsize, theta = theta, seed = seed)
        self.sampler_name = "GIST-Virial-Biased"

        self.switch_limit = switch_limit
        self.segment_length = segment_length
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
        theta0 = theta
        theta_star = theta
        theta_prop = theta
        #rho_star = rho
        #rho_prop = rho

        v = self.virial(theta, rho)
        sv = np.sign(v)
        H0 = self.log_joint(theta, rho)

        lsw_star = -np.inf
        lsw_prop = -np.inf

        lsw_segment = -np.inf

        steps_stop = 0
        steps_prop = 0
        steps_star = 0

        switches = 0
        switches_star = 0
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

            lsw_denom = lsw
            # track total energy
            lsw = np.logaddexp(lsw, delta)

            # multioulli: sample within segment
            lsw_segment = np.logaddexp(lsw_segment, delta)
            log_alpha = delta - lsw_segment
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_star = theta
                #rho_star = rho
                lsw_star = lsw
                steps_star = steps_stop
                switches_star = switches

            # biased: sample segment
            if steps_stop % self.segment_length == 0:
                log_beta = lsw_segment - lsw_denom
                if np.log(self.rng.uniform()) < np.minimum(0.0, log_beta):
                    theta_prop = theta_star
                    #rho_prop = rho_star
                    lsw_prop = lsw_star
                    steps_prop = steps_star
                    switches_prop = switches_star
                lsw_segment = -np.inf

                if switches >= self.switch_limit - self.switches_passed:
                    break

        self.steps += steps_stop
        if forward:
            self.switches_passed = switches_prop
            self.update_forward_steps(steps_stop)
            self.update_stopping_steps(steps_stop)
            self.update_proposal_steps(steps_prop)

        return theta_prop, lsw_prop, lsw

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)

            self.prepare_forward_pass()
            theta_star, lsw_star, FW = self.trajectory(theta, rho, forward = True)
            _, _, BW  = self.trajectory(theta, -rho, lsw = lsw_star)
            BW = self.logsubexp(BW, 0.0) # don't double count theta0

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
