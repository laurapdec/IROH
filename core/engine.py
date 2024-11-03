# core/engine.py

import time
import numpy as np
from tqdm import tqdm
from data_io.input_handler import InputHandler
from particles.particle_manager import ParticleManager
from fluid_solver.solver_interface import FluidSolverInterface
from micromixing.adaptive_micromixing import AdaptiveMicromixingModel
from tensor_utils.tensor_calculus import TensorCalculus
from chemistry.kinetics import ChemicalKinetics
from monte_carlo.monte_carlo_simulation import MonteCarloSimulation
from data_io.output_handler import LatexDataExporter

from micromixing.iem_model import IEMModel
from micromixing.curl_model import CurlModel
from micromixing.modified_curl_model import ModifiedCurlModel
from micromixing.adaptive_micromixing import AdaptiveMicromixingModel

class SimulationEngine:
    def __init__(self, config):
        # Store configuration and initialize timing
        self.config = config
        self.time = 0.0  # Simulation start time
        self.start_time = time.time()  # Capture start time for computational timing
        self.time_step = config['time_step']
        self.total_time = config['total_time']
        self.num_steps = int(self.total_time / self.time_step)
        self.current_step = 0
        self.last_export_time = 0.0  # Initialize export timing
        
        # Set up simulation label based on config values
        composition = config['initial_conditions']['composition']
        composition_label = '_'.join([f"{species}{int(frac*100)}" for species, frac in composition.items()])
        pressure = config['initial_conditions']['pressure']
        temperature = config['initial_conditions']['temperature']
        micromixing_model = config['micromixing_model']
        self.simulation_label = f"{composition_label}_P_{pressure:.0f}_T_{temperature:.0f}_{micromixing_model}"
        
        # Initialize LatexDataExporter with export directory
        export_directory = config.get('export_directory', 'latex_input')
        self.data_exporter = LatexDataExporter(self.simulation_label, export_directory=export_directory)
        
        # Set export interval and single-point export interval
        self.export_interval = config.get('export_interval', 0.1)
        self.single_point_export_interval = config.get('single_point_export_interval', 10)  # Default to every 10 steps
        
        # Initialize other components for simulation
        self.particle_manager = ParticleManager(config)
        self.fluid_solver = FluidSolverInterface(config)
        self.tensor_calculus = TensorCalculus(config)
        self.chemistry = ChemicalKinetics(config)
        self.monte_carlo = MonteCarloSimulation(config)

        # Initialize micromixing model based on config
        model_type = config.get("micromixing_model", "adaptive")
        if model_type == "iem":
            self.micromixing_model = IEMModel(config)
        elif model_type == "curl":
            self.micromixing_model = CurlModel()
        elif model_type == "modified_curl":
            self.micromixing_model = ModifiedCurlModel(config)
        else:
            self.micromixing_model = AdaptiveMicromixingModel(config)


    def collect_data(self):
        """Collects and exports continuous and single-point metrics once per interval."""

        # Continuous metrics - export once per interval
        if self.time - self.last_export_time >= self.export_interval:
            particles = self.particle_manager.particles
            scalar_variance = self.compute_scalar_variance('temperature')
            mean_temperature = self.compute_mean_scalar('temperature')
            rms_temperature = self.compute_rms_scalar('temperature')
            co_concentration = self.compute_mean_scalar('CO')
            
            # Prepare data for exports
            variance_data = [(self.time, scalar_variance)]
            mean_temp_data = [(p.position[0], p.properties['temperature']) for p in particles]
            rms_temp_data = [(p.position[1], p.properties['temperature']) for p in particles]
            co_concentration_data = [(p.position[0], p.properties.get('CO', np.nan)) for p in particles]

            # Export data
            self.data_exporter.export_scalar_variance_decay(variance_data)
            self.data_exporter.export_mean_temperature_profiles(mean_temp_data)
            self.data_exporter.export_rms_temperature_fluctuations(rms_temp_data)
            self.data_exporter.export_mean_co_concentration(co_concentration_data)
            
            self.last_export_time = self.time  # Update last export time

        # Single-point metrics - only export/update when the simulation completes or for specific intervals
        if self.time >= self.total_time or self.current_step % self.single_point_export_interval == 0:
            total_computational_time = time.time() - self.start_time
            particle_count_info = self.particle_manager.total_particle_count()
            
            # Append single-point metrics
            self.data_exporter.append_single_data_point("computational_times.dat", "Total Computational Time", total_computational_time)
            self.data_exporter.append_single_data_point("particle_count.dat", "Total Particle Count", particle_count_info)

    def transport_and_mix_particles(self):
        self.particle_manager.move_particles(self.time_step, self.fluid_solver)

        mean_properties = self.particle_manager.mean_scalar_values()

        # Apply the micromixing model to each pair or individual particle based on model type
        for particle in self.particle_manager.particles:
            position = particle.position
            strain_tensor = self.tensor_calculus.compute_rate_of_strain(position, self.fluid_solver)
            self.micromixing_model.apply_mixing(particle, strain_tensor, mean_properties)


    def run(self):
        print("Starting simulation...")
        start_time = time.time()
        with tqdm(
            total=self.num_steps,
            desc=f't = {self.time:.2f}s)',
            unit='step',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - Simulated Time: {postfix}'
        ) as pbar:
            while self.time < self.total_time:
                self.update_fluid_field()
                self.transport_and_mix_particles()
                self.process_reactions()
                self.collect_data()  # Collect all necessary data
                self.time += self.time_step
                self.current_step += 1
                pbar.set_postfix_str(f"{self.time:.2f}s")
                pbar.update(1)
        print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds.")

    def update_fluid_field(self):
        self.fluid_solver.update_flow_field(self.time)

    def transport_and_mix_particles(self):
        self.particle_manager.move_particles(self.time_step, self.fluid_solver)
        mean_properties = self.particle_manager.mean_scalar_values()
        for particle in self.particle_manager.particles:
            S = self.tensor_calculus.compute_rate_of_strain(particle.position, self.fluid_solver)
            self.micromixing_model.apply_mixing(particle, S, mean_properties)

    def process_reactions(self):
        self.chemistry.react_particles(self.particle_manager.particles)

    def compute_scalar_variance(self, scalar_name):
        values = [p.properties[scalar_name] for p in self.particle_manager.particles]
        mean_val = sum(values) / len(values)
        return sum((val - mean_val) ** 2 for val in values) / len(values)

    def compute_mean_scalar(self, scalar_name):
        values = [p.properties[scalar_name] for p in self.particle_manager.particles]
        return sum(values) / len(values)

    def compute_rms_scalar(self, scalar_name):
        values = [p.properties[scalar_name] for p in self.particle_manager.particles]
        mean_val = sum(values) / len(values)
        return (sum((val - mean_val) ** 2 for val in values) / len(values)) ** 0.5

