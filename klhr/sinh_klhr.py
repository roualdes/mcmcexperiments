import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from mcmc import MCMCBase
from bsmodel import BSModel

from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss

import scipy.special as sp
import scipy.stats as st
import cmdstanpy as csp
import numpy as np

class SINHKLHR(MCMCBase):
    def __init__(self, bsmodel, theta = None, seed = None,
                 N = 16, K = 10,
                 clip_trig = 650, tol_d = 1e-10,
                 clip_grad = 1e6, tol_grad = 1e12):
        super().__init__(bsmodel, -1, theta = theta, seed = seed)

        self.N = N
        self.x, self.w = hermgauss(self.N)
        self.K = K
        self.clip_trig = clip_trig
        self.clip_grad = clip_grad
        self.tol_grad = tol_grad
        self.tol_d = tol_d

        self._draw = 0
        self.acceptance_probability = 0
        self.min_failure_rate = 0

        # constants
        self.m = np.zeros(self.D)
        self.S = np.eye(self.D)
        self.sqrt2 = np.sqrt(2)
        self.log2 = np.log(2)
        self.log2pi = np.log(2 * np.pi)
        self.invsqrtpi = 1 / np.sqrt(np.pi)

    def _random_direction(self):
        rho = self.rng.multivariate_normal(self.m, self.S)
        return rho / np.linalg.norm(rho)

    def _to_rho(self, x, rho, origin):
        return x * rho + origin

    def _unpack(self, eta):
        m = eta[0]
        s = np.exp(eta[1]) + self.tol_d
        d = np.exp(eta[2]) + self.tol_d
        e = eta[3]
        return m, s, d, e

    def _T(self, x, eta):
        m, s, d, e = self._unpack(eta)
        return m + s * self._sinh_aed(x, eta)

    def _T_inv(self, x, eta):
        m, s, d, e = self._unpack(eta)
        z = (x - m) / s
        asinh_z = np.arcsinh(z)
        inside = d * (asinh_z - e)
        return np.sinh(inside)

    def _CDF(self, x, eta):
        t_inv = self._T_inv(x, eta)
        return sp.ndtr(t_inv)

    def _CDF_inv(self, x, eta):
        phi_inv = sp.ndtri(x)
        return self._T(phi_inv, eta)

    def _overrelaxed_proposal(self, eta):
        m, s, d, e = self._unpack(eta)
        K = self.K
        u = self._CDF(np.array([0]), eta)
        r = st.binom(K, u).rvs()
        up = 0
        if r > K - r:
            v = st.beta(K - r + 1, 2 * r - K).rvs()
            up = u * v
        elif r < K - r:
            v = st.beta(r + 1, K - 2 * r).rvs()
            up = 1 - (1 - u) * v
        elif r == K - r:
            up = u
        return self._CDF_inv(up, eta)

    def _logq(self, x, eta):
        m, s, d, e = self._unpack(eta)
        z = (x - m) / s
        asinhz = np.arcsinh(z)
        dae = d * asinhz - e
        abs_dae = np.abs(dae)
        out = eta[2] - eta[1] - self.log2
        out -= 0.5 * (self.log2pi + np.log1p(z * z) + 0.5 * (np.cosh(2 * dae) - 1))
        out += abs_dae + np.log1p(np.exp(-2 * abs_dae))
        return out

    def _sinh_aed(self, x, eta):
        _, _, d, e = self._unpack(eta)
        y = np.clip((np.arcsinh(x) + e) / d, -self.clip_trig, self.clip_trig)
        return np.sinh(y)

    def _cosh_aed(self, x, eta):
        _, _, d, e = self._unpack(eta)
        y = np.clip((np.arcsinh(x) + e) / d, -self.clip_trig, self.clip_trig)
        return np.cosh(y)

    def _logp_grad(self, x):
        f, g = self.model.log_density_gradient(x)
        g = np.clip(g, -self.clip_grad, self.clip_grad)
        ng = np.linalg.norm(g)
        if ng > self.tol_grad:
            g *= self.tol_grad / (ng + self.tol_d)
        return f, g

    def _grad_T(self, x, eta):
        m, s, d, e = self._unpack(eta)
        grad = np.zeros(4)
        grad[0] = 1
        asinhx = np.arcsinh(x)
        invd = 1 / d
        aed = (asinhx + e) * invd
        grad[1] = s * self._sinh_aed(x, eta)
        coshaed = self._cosh_aed(x, eta)
        grad[2] = -s * coshaed * aed
        grad[3] = s * coshaed * invd
        return grad

    def _log_cosh_asinh(self, x):
        return 0.5 * np.log1p(x * x)

    def _log_sech_aed(self, x, eta):
        m, s, d, e = self._unpack(eta)
        aed = (np.arcsinh(x) + e) / d
        return -np.abs(aed) - np.log1p(np.exp(-2 * np.abs(aed))) + np.log(2)

    def _log_abs_jac(self, x, eta):
        out = self._log_sech_aed(x, eta)
        out += eta[2] - eta[1]
        return out

    def _grad_log_abs_jac(self, x, eta):
        m, s, d, e = self._unpack(eta)
        invd = 1 / d
        grad = np.zeros(4)
        grad[1] = -1
        aed = (np.arcsinh(x) + e) * invd
        taed = np.tanh(aed)
        grad[2] = 1 + taed * aed
        grad[3] = -taed * invd
        return grad

    def _L(self, eta, rho):
        out = 0.0
        grad = np.zeros(4)
        m, s, d, e = self._unpack(eta)
        invd = 1 / d
        for xn, wn in zip(self.x, self.w):
            y = self.sqrt2 * xn
            log_abs_jac = self._log_abs_jac(y, eta)
            t = self._T(y, eta)
            xi = self._to_rho(t, rho, self.theta)
            logp, grad_logp = self._logp_grad(xi)
            out += wn * (log_abs_jac - logp)
            grad_log_abs_jac = self._grad_log_abs_jac(y, eta)
            grad_T = self._grad_T(y, eta)
            grad += wn * (grad_log_abs_jac - grad_logp.dot(rho) * grad_T)
        out *= self.invsqrtpi
        grad *= self.invsqrtpi
        return out, grad

    def fit(self, rho):
        o = minimize(self._L,
                     self.rng.normal(size = 4) * 0.1,
                     args = (rho,),
                     jac = True,
                     method = "BFGS")
        # self.min_failure_rate += (o.success - self.min_failure_rate) / self._draw
        return o.x # TODO consider returning o.success

    def _metropolis_step(self, eta, rho):
        # zp = self._T(self.rng.normal(size = 1), eta)
        zp = self._overrelaxed_proposal(eta)
        thetap = self._to_rho(zp, rho, self.theta)

        a = self.model.log_density(thetap) - self.model.log_density(self.theta)
        a += self._logq(0, eta)
        a -= self._logq(zp, eta)

        accept = np.log(self.rng.uniform()) < np.minimum(0, a)
        if accept:
            # print("accepted")
            self.theta = thetap

        d = accept - self.acceptance_probability
        self.acceptance_probability += d / self._draw

        return self.theta

    def draw(self):
        self._draw += 1
        rho = self._random_direction()
        etakl = self.fit(rho)
        theta = self._metropolis_step(etakl, rho)
        return theta
