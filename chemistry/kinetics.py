# chemistry/kinetics.py

import cantera as ct
import numpy as np

class ChemicalKinetics:
    def __init__(self, config):
        self.config = config
        self.mechanism_file = config['mechanism_file']
        self.load_mechanism()
        self.time_step = config['time_step']

    def load_mechanism(self):
        """
        Load and parse the chemical mechanism file using Cantera.
        """
        try:
            # Create a Cantera Solution object
            self.gas = ct.Solution(self.mechanism_file)
        except Exception as e:
            raise IOError(f"Error loading chemical mechanism: {e}")

    def react_particles(self, particles):
        for particle in particles:
            # Ensure only valid species are included in the composition
            composition = {
                species: particle.properties.get(species, 0.0)
                for species in self.gas.species_names
            }

            # Normalize composition to sum to 1 if the total is greater than zero
            total_composition = sum(composition.values())
            if total_composition > 0:
                composition = {k: v / total_composition for k, v in composition.items()}

            # Extract temperature and pressure
            temperature = particle.properties.get('temperature', 300.0)
            pressure = particle.properties.get('pressure', ct.one_atm)

            # Check for NaN values in temperature, pressure, or composition
            if np.isnan(temperature) or np.isnan(pressure) or any(np.isnan(value) for value in composition.values()):
                raise ValueError("Invalid initial conditions: temperature, pressure, or composition contains NaN values.")

            # Set the state of the gas object
            try:
                self.gas.TPY = temperature, pressure, composition
            except ct.CanteraError as e:
                raise RuntimeError(f"Failed to set state for gas: {e}")

            # Integrate the reactor over the time step
            reactor = ct.IdealGasConstPressureReactor(self.gas)
            sim = ct.ReactorNet([reactor])
            sim.advance(self.time_step)

            # Update particle properties with new state
            particle.properties['temperature'] = reactor.T
            particle.properties['pressure'] = reactor.thermo.P
            for i, species in enumerate(self.gas.species_names):
                particle.properties[species] = reactor.thermo.Y[i]