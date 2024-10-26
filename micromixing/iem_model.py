# micromixing/iem_model.py

class IEMModel:
    def __init__(self, config):
        self.mixing_constant = config.get('micromixing_constant', 1.0)
        self.delta_G = config.get('delta_G', 1.0)  # Assuming this parameter is defined in config

    def apply_mixing(self, particle, strain_tensor, mean_properties):
        # Compute Omega_m based on the strain tensor
        Gamma = strain_tensor.trace()  # Example of computing a general quantity Gamma
        Omega_m = (self.mixing_constant * (Gamma + strain_tensor.norm())) / (self.delta_G ** 2)

        # Apply the mixing model to each scalar property
        for prop, mean_val in mean_properties.items():
            particle.properties[prop] += -Omega_m * (particle.properties[prop] - mean_val) * particle.time_step
