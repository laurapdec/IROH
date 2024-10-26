# particles/particle.py

import numpy as np

class Particle:
    def __init__(self, position, properties):
        self.position = np.array(position)
        self.properties = properties.copy()
        self.velocity = np.zeros(3)

    def update_position(self, displacement):
        self.position += displacement

    def update_properties(self, new_properties):
        self.properties.update(new_properties)
