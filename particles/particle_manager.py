# particles/particle_manager.py

import numpy as np
import cantera as ct
from particles.particle import Particle

class ParticleManager:
    def __init__(self, config):
        self.config = config
        self.diffusivity = config.get('diffusivity', 1e-5)
        
        # Initialize Cantera gas object first
        self.gas = ct.Solution(config['mechanism_file'])
        
        # Now initialize particles after self.gas is defined
        self.particles = self.initialize_particles()

    def initialize_particles(self):
        initial_particles = []
        num_particles = self.config.get('num_particles', 100)
        
        # Use the initial composition specified in the configuration
        initial_composition = self.config.get('initial_conditions', {}).get('composition', {})
        
        # Set a default composition if the initial composition is empty or zero
        if sum(initial_composition.values()) == 0:
            initial_composition = {
                'CH4': 0.095,   # Adjust based on the mechanism and conditions
                'O2': 0.21,
                'N2': 0.695
            }
        
        temperature = self.config['initial_conditions'].get('temperature', 300.0)
        pressure = self.config['initial_conditions'].get('pressure', ct.one_atm)

        # Create full composition dictionary with all species in the mechanism
        full_composition = {species: initial_composition.get(species, 0.0) for species in self.gas.species_names}

        for _ in range(num_particles):
            position = self.random_initial_position()
            properties = {
                'temperature': temperature,
                'pressure': pressure,
                **full_composition  # Set species mass fractions
            }
            particle = Particle(position, properties)
            initial_particles.append(particle)
        return initial_particles
    
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

    def mean_scalar_values(self):
        num_particles = len(self.particles)
        scalar_sums = {}
        for particle in self.particles:
            for scalar, value in particle.properties.items():
                scalar_sums[scalar] = scalar_sums.get(scalar, 0.0) + value

        mean_values = {scalar: total / num_particles for scalar, total in scalar_sums.items()}
        return mean_values

    def random_initial_position(self):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        z = np.random.uniform(0, 1)
        return [x, y, z]


    def total_particle_count(self):
        """Returns the total count of particles managed."""
        return len(self.particles)
