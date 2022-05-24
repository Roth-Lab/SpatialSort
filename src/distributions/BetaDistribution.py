from scipy import stats


class BetaDistribution:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def sample(self):
        return stats.beta.rvs(self.a, self.b)

    def log_density(self, x):
        return stats.beta.logpdf(x, self.a, self.b)
