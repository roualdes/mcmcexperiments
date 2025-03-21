import numpy as np

class MCMCBase():
    def __init__(self, model, stepsize, theta = None, seed = None):
        self.model = model
        self.D = self.model.dim()
        self.stepsize = stepsize

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        if theta is None:
            self.theta = self.rng.normal(size = self.D)
        else:
            self.theta = theta

    def __iter__(self):
        return self

    def __next__(self):
        return self.draw()

    def log_density(self, theta):
        try:
            return self.model.log_density(theta)
        except Exception as e:
            return np.NINF

    def sample(self, M):
        D = self.D
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0, :] = self.theta;
        for m in range(1, M):
            thetas[m, :] = self.draw()
        return thetas

    def sample_constrained(self, M):
        D = self.D
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0, :] = self.model.constrain(self.theta)
        mean_proposal_steps = 0
        for m in range(1, M):
            # print(f"draw = {m}")
            theta_m = self.draw()
            thetas[m, :] = self.model.constrain(theta_m)
        return {"thetas": thetas,
                "steps": self.steps,
                "backward_proportion": self.backward_proportion,
                "mean_proposal_steps": self.mean_proposal_steps,
                "forward_steps": np.array(self.forward_steps) / M,
                "acceptance_rate": self.acceptance_probability}
