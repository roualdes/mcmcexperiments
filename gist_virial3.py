import numpy as np
import scipy as sp

import hmc
import traceback

from tools import logsubexp

class GISTV3(hmc.HMCBase):
    """Virial GIST

    Propose from foward trajectory and balance with backward
    trajectory.  switch_limit can be greater than 1. Proposals
    can be biased towards the end of the trajectory.

    """
    def __init__(self, model, stepsize,
                 theta = None, seed = None,
                 switch_limit = 1,
                 biased = True, **kwargs):
        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = "GIST-Virial-3"
        if theta is None:
            self.theta = self.rng.normal(size = self.D)
        else:
            self.theta = theta
        self.switch_limit = switch_limit
        self.biased = biased

        self.steps = 0
        self.prop_accepted = 0.0
        self.draws = 0
        self.divergences = 0

    def virial(self, theta, rho):
        return 2 * rho.dot(theta)

    def trajectory(self, theta, rho, lsw = -np.inf, switch_discount = 0):
        theta0 = theta
        theta_star = theta
        rho_star = rho
        v = self.virial(theta, rho)
        sv = np.sign(v)
        H_0 = self.log_joint(theta, rho)
        hs = [H_0]
        switches = 0
        steps = 0
        lsw_star = lsw
        switches_passed = 0
        while True:
            theta, rho = self.leapfrog_step(theta, rho)
            H = self.log_joint(theta, rho)
            if np.abs(H_0 - H) > 50.0:
              self.divergences += 1
              break

            v = self.virial(theta, rho)
            if sv * np.sign(v) < 0:
                switches += 1
                sv = np.sign(v)

            if switches >= self.switch_limit - switch_discount:
                break

            H = self.log_joint(theta, rho)
            d = 0.0
            if self.biased:
                # if self.biased == "switch":
                d = np.log(switches + 1)
                # elif self.biased == "distance":
                #     d = np.log(np.sum((theta - theta0) ** 2))
                # else:
                #     print("need specify biasing strategy")
            H += d

            lsw = np.logaddexp(lsw, H - H_0)
            log_alpha = H - H_0 - lsw
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_star = theta
                rho_star = rho
                switches_passed = switches
                lsw_star = lsw

            steps += 1

        self.steps += steps
        return theta_star, rho_star, lsw_star, lsw, switches_passed

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)
            H_0 = self.log_joint(theta, rho)

            theta_star, rho_star, lsw_star, FW, switches_passed = self.trajectory(theta, rho)
            H_star = self.log_joint(theta_star, rho_star)
            # _, _, _, BW, _ = self.trajectory(theta_star, -rho_star)

            lsw_star = np.logaddexp(lsw_star, 0.0)
            _, _, _, BW, _ = self.trajectory(theta, -rho,
                                              lsw = lsw_star,
                                              switch_discount = switches_passed)
            BW = logsubexp(BW, H_star - H_0)

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
