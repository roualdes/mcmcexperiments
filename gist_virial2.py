import numpy as np
import scipy as sp

import hmc
import traceback

class GISTV2(hmc.HMCBase):
    """Virial GIST

    Propose from foward trajectory and balance with backward
    trajectory.  switch_limit can be greater than 1. Proposals
    can be biased towards the end of the trajectory.

    No numerical efficiencies implemented.

    """
    def __init__(self, model, stepsize,
                 theta = None, seed = None,
                 switch_limit = 2,
                 biased = True, **kwargs):
        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = "GIST-Virial-2"
        if theta is None:
            self.theta = self.rng.normal(size = self.D)
        else:
            self.theta = theta
        self.switch_limit = switch_limit
        self.biased = biased

        self.steps = 0
        self.prop_accepted = 0.0
        self.draws = 0

    def virial(self, theta, rho):
        return 2 * rho.dot(theta)

    def trajectory(self, theta, rho):
        theta0 = theta
        v = self.virial(theta, rho)
        sv = np.sign(v)
        hs = [self.log_joint(theta, rho)]
        switches = 0
        steps = 0
        while True:
            theta, rho = self.leapfrog_step(theta, rho)

            v = self.virial(theta, rho)
            if sv * np.sign(v) < 0:
                switches += 1
                sv = np.sign(v)

            if switches >= self.switch_limit:
                break

            H = self.log_joint(theta, rho)
            if self.biased:
                if self.biased == "switch":
                    d = np.log(switches + 1)
                elif self.biased == "distance":
                    d = np.log(np.sum((theta - theta0) ** 2))
                else:
                    print("need specify biasing strategy")

                hs.append(H + d)
            else:
                hs.append(H)
            steps += 1

        return steps, hs

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)

            F, FHs = self.trajectory(theta, rho)
            self.steps += F
            FW = sp.special.logsumexp(FHs)
            N = self.rng.choice(F + 1, p = np.exp(FHs - FW))

            theta_star, rho_star = self.leapfrog(theta, rho, N)
            self.steps += N

            B, BHs = self.trajectory(theta_star, -rho_star)
            self.steps += B
            BW = sp.special.logsumexp(BHs)

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
