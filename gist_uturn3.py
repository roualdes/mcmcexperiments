import numpy as np

import hmc
import traceback

class GISTU3(hmc.HMCBase):
    """GIST UTURN

    Online Multinoulli sampling during forward trajectory.  Proposals
    are biased towards the end of the trajectory in segments
    consisting of segment_length steps.  Biasing analogous to Stan.

    """
    def __init__(self, model, stepsize, theta = None, seed = None,
                 segment_length = 1, **kwargs):

        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = "GIST-UTURN-3"

        if theta is not None:
            self.theta = theta

        self.segment_length = segment_length
        self.steps = 0
        self.prop_accepted = 0.0
        self.draws = 0
        self.divergences = 0
        self.forward_steps = []
        self.mean_proposal_steps = 0

    def logsubexp(self, a, b):
        if a > b:
            return a + np.log1p(-np.exp(b - a))
        elif a < b:
            return b + np.log1p(-np.exp(a - b))
        else:
            return np.inf

    def trajectory(self, theta, rho, lsw = 0.0):
        theta0 = theta
        last_distance = 0
        theta_star = theta
        rho_star = rho
        H_0 = self.log_joint(theta, rho)
        uturn = False
        steps = 0
        lsw_segment = -np.inf
        lsw_star = -np.inf
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

            # check uturn
            distance = np.sum((theta - theta0) ** 2)
            if distance <= last_distance:
                uturn = True
            last_distance = distance

            # sample new state
            lsw_segment = np.logaddexp(lsw_segment, delta)
            lsw = np.logaddexp(lsw, delta)
            log_alpha = delta - lsw
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_star = theta
                rho_star = rho
                lsw_star = H
                proposal_steps = steps

            # sample from last segment and start new segment
            if end_segment:
                log_beta = lsw - lsw_segment
                if np.log(self.rng.uniform()) < np.minimum(0.0, log_beta):
                    theta_prop = theta_star
                    rho_prop = rho_star
                    proposal_steps = steps
                    lsw_prop = lsw
                lsw_segment = -np.inf

            if uturn and end_segment:
                break

        self.steps += steps
        return theta_star, rho_star, lsw, lsw_star, proposal_steps

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)
            H_0 = self.log_joint(theta, rho)

            theta_star, _, FW, lsw_star, proposal_steps = self.trajectory(theta, rho)
            F = self.steps

            self.forward_steps.append(F)
            self.mean_proposal_steps += (proposal_steps - self.mean_proposal_steps) / self.draws

            _, _, BW, _, _ = self.trajectory(theta, -rho)

            B = self.steps - F

            # account for sub-uturns
            if not(1 <= proposal_steps and proposal_steps <= B):
                return self.theta

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
