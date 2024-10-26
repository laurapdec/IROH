# tensor_utils/tensor_calculus.py

import numpy as np

class TensorCalculus:
    def __init__(self, config):
        self.config = config
        # Any additional initialization

    def compute_rate_of_strain(self, position, fluid_solver):
        """
        Compute the rate-of-strain tensor at a given position.
        :param position: numpy array of shape (3,)
        :param fluid_solver: an instance of FluidSolverInterface
        :return: numpy array of shape (3, 3) representing the rate-of-strain tensor
        """
        # Compute velocity gradients at the position
        velocity_gradients = self.compute_velocity_gradients(position, fluid_solver)

        # Compute the rate-of-strain tensor
        S = 0.5 * (velocity_gradients + velocity_gradients.T)

        return S

    def compute_velocity_gradients(self, position, fluid_solver):
        """
        Compute the velocity gradient tensor at a given position.
        :param position: numpy array of shape (3,)
        :param fluid_solver: an instance of FluidSolverInterface
        :return: numpy array of shape (3, 3) representing the velocity gradient tensor
        """
        delta = 1e-5  # Small perturbation for finite differences

        # Initialize gradient tensor
        grad_u = np.zeros((3, 3))

        # Perturbation along x
        dx = np.array([delta, 0, 0])
        u_plus = fluid_solver.get_velocity_at(position + dx)
        u_minus = fluid_solver.get_velocity_at(position - dx)
        grad_u[:, 0] = (u_plus - u_minus) / (2 * delta)

        # Perturbation along y
        dy = np.array([0, delta, 0])
        v_plus = fluid_solver.get_velocity_at(position + dy)
        v_minus = fluid_solver.get_velocity_at(position - dy)
        grad_u[:, 1] = (v_plus - v_minus) / (2 * delta)

        # Perturbation along z
        dz = np.array([0, 0, delta])
        w_plus = fluid_solver.get_velocity_at(position + dz)
        w_minus = fluid_solver.get_velocity_at(position - dz)
        grad_u[:, 2] = (w_plus - w_minus) / (2 * delta)

        return grad_u
