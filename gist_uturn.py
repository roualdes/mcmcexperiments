import numpy as np
import scipy as sp

import hmc
import traceback

class GISTU(hmc.HMCBase):
    def __init__(self, model, stepsize,
                 theta = None, seed = None,
                 sign_switch_limit = 1, **kwargs):
        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = "GIST-Uturn"
        if theta is None:
            self.theta = self.rng.normal(size = self.D)
        else:
            self.theta = theta
        self.prop_accepted = 0.0
        self.draws = 0
        self.sign_switch_limit = sign_switch_limit
        self.steps = 0

    def apogee(self, theta, rho):
        _, g = self.model.log_density_gradient(theta)
        return g.dot(rho)

    def trajectory(self, theta, rho):
        theta0 = theta
        last_distance = 0.0
        steps = 0
        while True:
            theta, rho = self.leapfrog_step(theta, rho)
            distance = np.sum((theta - theta0) ** 2)
            if distance <= last_distance:
                return steps
            last_distance = distance
            steps += 1

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)
            H_0 = self.log_joint(theta, rho)

            F = self.trajectory(theta, rho)
            self.steps += F
            N = self.rng.integers(1, F + 1)

            theta_star, rho_star = self.leapfrog(theta, rho, N)
            self.steps += N

            B = self.trajectory(theta_star, -rho_star)
            self.steps += B
            H_star = self.log_joint(theta_star, rho_star)

            if not(1 <= N and N <= B):
                return self.theta

            log_alpha = H_star - H_0 + np.log(F) - np.log(B)

            accepted = 0
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                self.theta = theta_star
                accepted = 1

            self.prop_accepted += (accepted - self.prop_accepted) / self.draws
        except Exception as e:
            traceback.print_exc()
            pass
        return self.theta
