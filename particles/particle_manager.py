# particles/particle_manager.py

import numpy as np
from particles.particle import Particle

class ParticleManager:
    def __init__(self, config):
        self.particles = []
        self.diffusivity = config.get('diffusivity', 1e-5)
        self.initialize_particles(config['initial_conditions'])

    def initialize_particles(self, initial_conditions):
        for condition in initial_conditions:
            particle = Particle(condition['position'], condition['properties'])
            self.particles.append(particle)

    def move_particles(self, time_step, fluid_solver):
        for particle in self.particles:
            velocity = fluid_solver.get_velocity_at(particle.position)
            stochastic_disp = self.get_stochastic_displacement(time_step)
            total_displacement = velocity * time_step + stochastic_disp
            particle.update_position(total_displacement)
            particle.velocity = velocity

    def get_stochastic_displacement(self, time_step):
        sigma = np.sqrt(2 * self.diffusivity * time_step)
        return np.random.normal(0, sigma, size=3)
