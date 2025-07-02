import numpy as np

import hmc
import traceback

class GISTUM(hmc.HMCBase):
    """GIST Uturn

    Sample along trajectory using multinoulli

    """
    def __init__(self, model, stepsize, theta = None, seed = None, **kwargs):

        super().__init__(model, stepsize, theta = theta, seed = seed)
        self.sampler_name = "GIST-Uturn-Multinoulli"

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

    def trajectory(self, theta, rho, forward = False):
        theta0 = theta
        theta_prop = theta
        rho_prop = rho

        steps_stop = 0
        steps_prop = 0

        uturn = False
        last_distance = 0

        H0 = self.log_joint(theta, rho)
        lsw = 0.0 # H0 - H0

        while True:
            steps_stop += 1

            theta, rho = self.leapfrog_step(theta, rho)
            H = self.log_joint(theta, rho)
            delta = H - H0
            if np.abs(-delta) > 50.0:
                self.divergences += 1
                break

            distance = np.sum((theta - theta0) ** 2)
            if distance <= last_distance:
                uturn = True
            last_distance = distance

            # track total energy
            lsw = np.logaddexp(lsw, delta)

            # multinoulli: sample within trajectory
            log_alpha = delta - lsw
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_prop = theta
                rho_prop = rho
                steps_prop = steps_stop

            if uturn:
                break

        self.steps += steps_stop
        if forward:
            self.update_forward_steps(steps_stop)
            self.update_stopping_steps(steps_stop)
            self.update_proposal_steps(steps_prop)

        return theta_prop, rho_prop, steps_prop, steps_stop, lsw

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)

            theta_star, rho_star, F, _, FW = self.trajectory(theta, rho, forward = True)

            _, _, _, B, BW = self.trajectory(theta_star, -rho_star)

            # account for sub-uturns
            if not(1 <= F and F <= B):
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
