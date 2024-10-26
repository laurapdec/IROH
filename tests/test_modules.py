# tests/test_modules.py

import unittest
from particles.particle import Particle
from particles.particle_manager import ParticleManager
# Import other modules as needed

class TestParticle(unittest.TestCase):
    def test_particle_initialization(self):
        position = [0.0, 0.0, 0.0]
        properties = {'temperature': 300.0}
        particle = Particle(position, properties)
        self.assertEqual(particle.position.tolist(), position)
        self.assertEqual(particle.properties, properties)

# Add more test cases for other modules

if __name__ == '__main__':
    unittest.main()
