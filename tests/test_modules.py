# tests/test_modules.py

import unittest
import h5py
import numpy as np
import os
import cantera as ct

from chemistry.kinetics import ChemicalKinetics
from particles.particle import Particle
from particles.particle_manager import ParticleManager
from fluid_solver.solver_interface import FluidSolverInterface
from tensor_utils.tensor_calculus import TensorCalculus
from micromixing.adaptive_micromixing import AdaptiveMicromixingModel

class TestParticle(unittest.TestCase):
    def test_particle_initialization(self):
        position = [0.0, 0.0, 0.0]
        properties = {'temperature': 300.0}
        particle = Particle(position, properties)
        self.assertEqual(particle.position.tolist(), position)
        self.assertEqual(particle.properties, properties)

class TestFluidSolverInterface(unittest.TestCase):
    def setUp(self):
        # Create a mock configuration
        self.config = {
            'flow_field_file': 'test_flow_field.h5',
            'flow_field_time_dependent': False
        }
        # Generate mock flow field data and save to HDF5 file
        self.create_mock_flow_field()

        # Initialize the fluid solver
        self.solver = FluidSolverInterface(self.config)

    def create_mock_flow_field(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 10)
        u = np.ones((10, 10, 10))  # Simple uniform flow in the x-direction
        v = np.zeros((10, 10, 10))
        w = np.zeros((10, 10, 10))

        with h5py.File('test_flow_field.h5', 'w') as f:
            f.create_dataset('x', data=x)
            f.create_dataset('y', data=y)
            f.create_dataset('z', data=z)
            f.create_dataset('u', data=u)
            f.create_dataset('v', data=v)
            f.create_dataset('w', data=w)

    def test_get_velocity_at(self):
        position = np.array([0.5, 0.5, 0.5])
        velocity = self.solver.get_velocity_at(position)
        expected_velocity = np.array([1.0, 0.0, 0.0])
        np.testing.assert_almost_equal(velocity, expected_velocity)

    def tearDown(self):
        # Clean up the mock flow field file
        os.remove('test_flow_field.h5')

class TestTensorCalculus(unittest.TestCase):
    def setUp(self):
        # Create a mock configuration
        self.config = {
            'flow_field_file': 'test_flow_field.h5',
            'flow_field_time_dependent': False
        }
        # Generate mock flow field data and save to HDF5 file
        self.create_mock_flow_field()

        # Initialize fluid solver and tensor calculus
        self.fluid_solver = FluidSolverInterface(self.config)
        self.tensor_calculus = TensorCalculus(self.config)

    def create_mock_flow_field(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        z = np.linspace(0, 1, 10)

        # Define a simple linear velocity field for testing: u = x, v = y, w = z
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        u = X
        v = Y
        w = Z

        with h5py.File('test_flow_field.h5', 'w') as f:
            f.create_dataset('x', data=x)
            f.create_dataset('y', data=y)
            f.create_dataset('z', data=z)
            f.create_dataset('u', data=u)
            f.create_dataset('v', data=v)
            f.create_dataset('w', data=w)

    def test_compute_rate_of_strain(self):
        position = np.array([0.5, 0.5, 0.5])
        S = self.tensor_calculus.compute_rate_of_strain(position, self.fluid_solver)

        # For the linear velocity field u=x, v=y, w=z, the velocity gradient tensor is the identity matrix
        expected_grad_u = np.eye(3)
        expected_S = 0.5 * (expected_grad_u + expected_grad_u.T)
        np.testing.assert_almost_equal(S, expected_S)

    def tearDown(self):
        # Clean up the mock flow field file
        os.remove('test_flow_field.h5')

class TestMicromixingModel(unittest.TestCase):
    def setUp(self):
        # Mock configuration
        self.config = {
            'time_step': 0.01,
            'micromixing_constant': 1.0,
            'diffusivity': 1e-5
        }
        # Initialize the micromixing model
        self.micromixing_model = AdaptiveMicromixingModel(self.config)
        # Create a particle
        self.particle = Particle(position=np.array([0.5, 0.5, 0.5]), properties={'temperature': 300.0})
        # Define mean properties
        self.mean_properties = {'temperature': 350.0}
        # Define a mock rate-of-strain tensor
        self.S = np.eye(3)  # Identity matrix

    def test_compute_micromixing_rate(self):
        micromixing_rate = self.micromixing_model.compute_micromixing_rate(self.S)
        expected_S_squared = np.sum(self.S**2)
        expected_chi = 2 * self.config['diffusivity'] * expected_S_squared
        expected_micromixing_rate = self.config['micromixing_constant'] * expected_chi
        self.assertAlmostEqual(micromixing_rate, expected_micromixing_rate)

    def test_mix_particle(self):
        initial_temperature = self.particle.properties['temperature']
        micromixing_rate = self.micromixing_model.compute_micromixing_rate(self.S)
        self.micromixing_model.mix_particle(self.particle, micromixing_rate, self.mean_properties)
        expected_temperature = initial_temperature - micromixing_rate * (initial_temperature - self.mean_properties['temperature']) * self.config['time_step']
        self.assertAlmostEqual(self.particle.properties['temperature'], expected_temperature)

class TestChemicalKinetics(unittest.TestCase):
    def setUp(self):
       # Mock configuration with a larger time step to allow for reaction progress
        self.config = {
            'mechanism_file': 'gri30.yaml',
            'time_step': 1e-3  # Increase time step temporarily
        }
        # Initialize the chemical kinetics module
        self.chemistry = ChemicalKinetics(self.config)
        # Create a particle with initial composition
        self.particle = Particle(
            position=[0.0, 0.0, 0.0],
            properties={
                'temperature': 1200.0,  # Increase initial temperature
                'pressure': ct.one_atm,
                'CH4': 0.5,
                'O2': 0.5,
                'N2': 0.0
            }
        )
        
    def test_load_mechanism(self):
        self.assertIsNotNone(self.chemistry.gas)
        self.assertIn('CH4', self.chemistry.gas.species_names)

    def test_react_particles(self):
        initial_CH4 = self.particle.properties['CH4']
        initial_temperature = self.particle.properties['temperature']

        self.chemistry.react_particles([self.particle])

        # After reaction, CH4 should decrease and temperature should increase (exothermic reaction)
        final_CH4 = self.particle.properties['CH4']
        final_temperature = self.particle.properties['temperature']

        self.assertLess(final_CH4, initial_CH4)
        self.assertGreater(final_temperature, initial_temperature)

    def tearDown(self):
        pass  # No cleanup needed

if __name__ == '__main__':
    unittest.main()