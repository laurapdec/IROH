# monte_carlo/monte_carlo_simulation.py

import numpy as np

class MonteCarloSimulation:
    def __init__(self, config):
        self.num_samples = config.get('num_samples', 1000)

    def perform_sampling(self, distribution):
        samples = distribution.rvs(size=self.num_samples)
        return samples
