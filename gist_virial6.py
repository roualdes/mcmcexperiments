import numpy as np

import hmc
import traceback

class GISTV6(hmc.HMCBase):
    """Virial GIST

    Propose from foward trajectory and balance with backward
    trajectory.  switch_limit can be greater than 1. Proposals are
    biased towards the end of the trajectory in segments consisting of
    segment_length steps.  Biasing analogous to Stan.  Trialing new biasing.

    This seems to work well, but I don't understand why. Checkpoint.

    """
    def __init__(self, model, stepsize, theta = None, seed = None,
                 switch_limit = 1, segment_length = 2, **kwargs):

        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = f"GIST-V6_{switch_limit}"

        if theta is not None:
            self.theta = theta

        self.segment_length = segment_length
        self.switch_limit = switch_limit

        self.steps = 0
        self.prop_accepted = 0.0
        self.draws = 0
        self.divergences = 0
        self.forward_steps = []
        self.mean_proposal_steps = 0

    def virial(self, theta, rho):
        return 2 * rho.dot(theta)

    def logsubexp(self, a, b):
        if a > b:
            return a + np.log1p(-np.exp(b - a))
        elif a < b:
            return b + np.log1p(-np.exp(a - b))
        else:
            return np.inf

    def trajectory(self, theta, rho, lsw = 0.0, switch_discount = 0):
        theta0 = theta
        theta_star = theta
        rho_star = rho
        theta_prop = theta
        rho_prop = rho
        v = self.virial(theta, rho)
        sv = np.sign(v)
        H_0 = self.log_joint(theta, rho)
        switches = 0
        steps = 0
        lsw_new = -np.inf
        lsw_old = 0.0
        lsw_prop = -np.inf
        switches_passed = 0
        end_segment = False
        proposal_steps = 0
        while True:
            steps += 1
            end_segment = steps % self.segment_length == 0

            # leapfrog step
            theta, rho = self.leapfrog_step(theta, rho)
            H = self.log_joint(theta, rho)
            delta = H - H_0
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

            # sample state within segment
            # lsw_new = np.logaddexp(lsw_new, delta)
            log_alpha = delta - lsw
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_star = theta
                rho_star = rho
                lsw_prop = lsw
                switches_passed = switches

            # sample segment and start new segment
            if end_segment:
                log_beta = lsw_new - lsw
                if np.log(self.rng.uniform()) < np.minimum(0.0, log_beta):
                    theta_prop = theta_star
                    rho_prop = rho_star
                    proposal_steps = steps
                    lsw_prop = lsw
                # lsw_old = lsw_new
                lsw_new = -np.inf
            else:
                lsw_new = np.logaddexp(lsw_new, delta)

            # if end_segment and reached switch limit, break
            if end_segment and switches >= self.switch_limit - switch_discount:
                break

        self.steps += steps
        return theta_star, rho_star, lsw_prop, lsw, switches_passed, proposal_steps
        # return theta_prop, rho_prop, lsw_prop, lsw, switches_passed, proposal_steps

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)
            H_0 = self.log_joint(theta, rho)

            theta_star, rho_star, lsw_star, FW, switches_passed, proposal_steps = self.trajectory(theta, rho)
            self.forward_steps.append(self.steps)
            self.mean_proposal_steps += (proposal_steps - self.mean_proposal_steps) / self.draws

            _, _, _, BW, _, _ = self.trajectory(theta, -rho,
                                                lsw = lsw_star,
                                                switch_discount = switches_passed)
            BW = self.logsubexp(BW, 0.0) # don't double count theta0

            # H_star - H_0 + (H_0 - BW) - (H_star - FW)
            log_alpha = FW - BW

            accepted = 0
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                accepted = 1
                self.theta = theta_star

            self.prop_accepted += (accepted - self.prop_accepted) / self.draws
        except Exception as e:
            traceback.print_exc()
            pass
        return self.theta
