# core/engine.py

import time
from tqdm import tqdm
from data_io.input_handler import InputHandler
from data_io.output_handler import OutputHandler
from particles.particle_manager import ParticleManager
from fluid_solver.solver_interface import FluidSolverInterface
from micromixing.adaptive_micromixing import AdaptiveMicromixingModel
from tensor_utils.tensor_calculus import TensorCalculus
from chemistry.kinetics import ChemicalKinetics
from monte_carlo.monte_carlo_simulation import MonteCarloSimulation

class SimulationEngine:
    def __init__(self, config):
        # Initialize components
        self.config = config
        self.input_handler = InputHandler(config)
        self.output_handler = OutputHandler(config)
        self.particle_manager = ParticleManager(config)
        self.fluid_solver = FluidSolverInterface(config)
        self.micromixing_model = AdaptiveMicromixingModel(config)
        self.tensor_calculus = TensorCalculus(config)
        self.chemistry = ChemicalKinetics(config)
        self.monte_carlo = MonteCarloSimulation(config)
        self.time = 0.0
        self.time_step = config['time_step']
        self.total_time = config['total_time']
        self.num_steps = int(self.total_time / self.time_step)
        self.current_step = 0

    def run(self):
        print("Starting simulation...")
        start_time = time.time()
        with tqdm(
            total=self.num_steps,
            desc='Simulation Progress',
            unit='step',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        ) as pbar:
            while self.time < self.total_time:
                self.update_fluid_field()
                self.transport_and_mix_particles()
                self.process_reactions()
                self.collect_data()
                self.time += self.time_step
                self.current_step += 1

                # Update the progress bar
                pbar.update(1)

        end_time = time.time()
        print(f"\nSimulation completed in {end_time - start_time:.2f} seconds.")
    
    def update_fluid_field(self):
        self.fluid_solver.update_flow_field(self.time)

    def transport_and_mix_particles(self):
        self.particle_manager.move_particles(self.time_step, self.fluid_solver)
        self.micromixing_model.apply_mixing(self.particle_manager.particles, self.tensor_calculus)

    def process_reactions(self):
        self.chemistry.react_particles(self.particle_manager.particles, self.time_step)

    def collect_data(self):
        self.output_handler.save_state(self.time, self.particle_manager.particles)
        # Additional data collection, such as scalar variance
        variance = self.compute_scalar_variance(self.particle_manager.particles, 'temperature')
        self.output_handler.save_scalar_variance(self.time, variance)

    def compute_scalar_variance(self, particles, scalar_name):
        scalar_values = [particle.properties[scalar_name] for particle in particles]
        mean_value = sum(scalar_values) / len(scalar_values)
        variance = sum((value - mean_value) ** 2 for value in scalar_values) / len(scalar_values)
        return variance
