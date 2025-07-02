import numpy as np
from mcmc import MCMCBase

class HMCBase(MCMCBase):
    def __init__(self, model, stepsize, seed = None, theta = None):
        super().__init__(model, stepsize, seed = seed, theta = theta)
        self.rho = self.rng.normal(size=self.D)

    def leapfrog_step(self, theta, rho):
        _, grad = self.model.log_density_gradient(theta)
        rho2 = rho + 0.5 * self.stepsize * grad
        theta2 = theta + self.stepsize * rho2
        _, grad = self.model.log_density_gradient(theta2)
        rho2 += 0.5 * self.stepsize * grad
        return theta2, rho2

    def leapfrog(self, theta, rho, numsteps):
        for _ in range(numsteps):
            theta, rho = self.leapfrog_step(theta, rho)
        return theta, rho

    def log_joint(self, theta, rho):
        try:
            return self.model.log_density(theta) - 0.5 * rho.dot(rho)
        except Exception as e:
            return -np.inf
