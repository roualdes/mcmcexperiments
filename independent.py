import numpy as np
import hmc
import traceback

class IM(hmc.HMCBase):
    def __init__(self, model, stepsize,
                 population_size = 1,
                 theta = None,
                 seed = None,
                 **kwargs):
        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = "Independent Metropolis"
        self.pop_size = population_size
        self.pop_thetas = np.zeros(shape = (self.pop_size, self.D))
        self.accepted = 0

        if theta is None:
            self.theta = self.rng.normal(size = self.D)
        else:
            self.theta = theta

        for ps in range(self.pop_size):
            self.pop_thetas[ps] = self.theta

    def proposal_density(self, theta_p, theta):
        stepsize = self.stepsize
        z = theta_p - theta
        return -0.5 * z.dot(z) / stepsize

    def draw(self):
        try:
            idx = self.rng.integers(self.pop_size)
            theta_old = np.copy(self.pop_thetas[idx])

            stepsize = self.stepsize

            for ps in range(self.pop_size):
                xi = self.rng.normal(size = self.D)
                self.pop_thetas[ps] = stepsize * xi

            theta_star = self.pop_thetas[idx]

            H1 = self.log_density(theta_star) + self.proposal_density(theta_old, 0.0)
            H0 = self.log_density(self.theta) + self.proposal_density(0.0, theta_old)

            log_alpha = H1 - H0
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                self.theta = theta_star
                self.accepted += 1
        except Exception as e:
            # traceback.print_exc()
            pass
        return self.theta
