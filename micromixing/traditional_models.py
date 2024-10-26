# micromixing/traditional_models.py

class IEMMicromixingModel:
    def __init__(self, config):
        self.micromixing_rate = config.get('micromixing_rate', 1.0)

    def apply_mixing(self, particles):
        for particle in particles:
            for scalar in particle.properties:
                fluctuation = particle.properties[scalar] - self.mean_scalar_value(scalar)
                particle.properties[scalar] -= self.micromixing_rate * fluctuation

    def mean_scalar_value(self, scalar_name):
        # Placeholder: Return a mean scalar value for mixing
        return 0.0  # Replace with appropriate calculation

# Additional traditional models can be implemented similarly
