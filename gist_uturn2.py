import numpy as np

import hmc
import traceback

class GISTU2(hmc.HMCBase):
    """GIST UTURN

    Online Multinoulli sampling during forward trajectory.

    """
    def __init__(self, model, stepsize, theta = None, seed = None, **kwargs):

        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = "GIST-UTURN-2"

        if theta is not None:
            self.theta = theta

        self.steps = 0
        self.forward_steps = []
        self.mean_proposal_steps = 0.0

        self.backward_proportion = 1

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

    def store_acceptance_probability(self, accepted):
        d = accepted - self.acceptance_probability
        self.acceptance_probability += d / self.draws


    def trajectory(self, theta, rho):
        theta0 = theta
        theta_prop = theta
        rho_prop = rho

        steps = 0
        steps_prop = 0

        uturn = False
        last_distance = 0

        H0 = self.log_joint(theta, rho)
        lsw = 0.0 # H0 - H0
        lsw_prop = -np.inf

        while True:
            steps += 1

            # leapfrog step
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
                lsw_prop = lsw
                steps_prop = steps

            if uturn:
                break

        self.steps += steps
        self.proposal_steps = steps_prop
        return theta_prop, lsw

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)
            H_0 = self.log_joint(theta, rho)

            # forward pass
            self.prepare_forward_pass()
            theta_star, FW = self.trajectory(theta, rho)
            F = self.steps

            # only for comparisons, otherwise unnecessary
            self.store_forward_steps()
            self.store_proposal_steps()

            # backward pass
            _, BW = self.trajectory(theta, -rho)
            B = self.steps - F

            # account for sub-uturns
            if not(1 <= self.proposal_steps and self.proposal_steps <= B):
                return self.theta

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
