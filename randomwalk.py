import numpy as np
import mcmc
import traceback

class RW(mcmc.MCMCBase):
    def __init__(self, model, stepsize, seed = None, theta = None):
        super().__init__(model, stepsize, seed = seed, theta = theta)
        self.sampler_name = "Random Walk"
        self.accepted = 0

    def proposal_density(self, thetap, theta):
        stepsize = self.stepsize
        z = (thetap - theta) / stepsize
        return -0.5 * z.dot(z)

    def draw(self):
        try:
            xi = self.rng.normal(size = self.D)
            theta_star = self.theta + xi * self.stepsize

            H1 = self.log_density(theta_star) + \
                self.proposal_density(self.theta, theta_star)

            H0 = self.log_density(self.theta) + \
                self.proposal_density(theta_star, self.theta)

            log_alpha = H1 - H0
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                self.theta = theta_star
                self.accepted += 1
        except Exception as e:
            # traceback.print_exc()
            pass
        return self.theta
