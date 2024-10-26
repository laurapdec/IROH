# micromixing/modified_curl_model.py

class ModifiedCurlModel:
    def __init__(self, config):
        self.alpha = config.get('alpha', 0.5)  # Mixing coefficient, typically between 0 and 1

    def apply_mixing(self, particle_a, particle_b):
        for prop in particle_a.properties:
            avg = (particle_a.properties[prop] + particle_b.properties[prop]) / 2
            particle_a.properties[prop] = (1 - self.alpha) * particle_a.properties[prop] + self.alpha * avg
            particle_b.properties[prop] = (1 - self.alpha) * particle_b.properties[prop] + self.alpha * avg
