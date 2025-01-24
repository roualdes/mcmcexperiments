import numpy as np
import hmc
import traceback

class MALA(hmc.HMCBase):
    def __init__(self, model, stepsize, theta = None, seed = None, **kwargs):
        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = "MALA"
        if theta is None:
            self.theta = self.rng.normal(size = self.D)
        else:
            self.theta = theta
        self.accepted = 0

    def proposal_density(self, theta_p, theta):
        _, g = self.model.log_density_gradient(theta)
        stepsize = self.stepsize
        z = theta_p - theta - stepsize * g
        return -0.25 * z.dot(z) / stepsize

    def draw(self):
        try:
            theta = self.theta
            stepsize = self.stepsize
            D = self.D
            xi = self.rng.normal(size = D)
            _, g = self.model.log_density_gradient(theta)
            theta_star = theta + stepsize * g + np.sqrt(2 * stepsize) * xi

            H1 = self.log_density(theta_star) + self.proposal_density(theta, theta_star)
            H0 = self.log_density(theta) + self.proposal_density(theta_star, theta)

            log_alpha = H1 - H0
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                self.theta = theta_star
                self.accepted += 1
        except Exception as e:
            pass
            # traceback.print_exc()
        return self.theta
