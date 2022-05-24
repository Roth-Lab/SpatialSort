import numpy as np
from scipy import stats


class SimplexProposalDistribution:

    def __init__(self, scale):
        self.current_value = None
        self.scale = scale

    def set_current_value(self, x):
        self.current_value = np.copy(x)

    def propose(self, from_x):
        proposal = np.copy(from_x)
        dim = proposal.shape[-1]

        # Sample a random value from norm for n-1, then for the nth value take 1 - sum
        sum = 0
        for i in np.arange(dim - 1):
            # values may exceed 1
            proposal[i] = stats.norm.rvs(loc=proposal[i], scale=self.scale)
            sum += proposal[i]
        proposal[dim - 1] = 1 - sum

        return proposal
