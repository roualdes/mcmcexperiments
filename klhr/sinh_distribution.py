import numpy as np

class SinhArcsinhDistribution():
    def __init__(self, m = 0.0, s = 0.0, d = 0.0, e = 0.0):
        self.m = m
        self.s = s
        self.d = d
        self.e = e

    def logpdf(self, x):
        m, s, d, e = self.m, self.s, self.d, self.e
        z = (x - m) / s
        out = -0.5 * np.log(2 * np.pi)
        out += -0.25 * (np.cosh(2 * d * np.arcsinh(z) - 2 * e) - 1)
        out += np.log(np.cosh(d * np.arcsinh(z) - e))
        out += np.log(d) - np.log(s)
        out += -0.5 * np.log(1 + z**2)
        return out

    def pdf(self, x):
        return np.exp(self.logpdf(x))