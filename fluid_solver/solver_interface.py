# fluid_solver/solver_interface.py

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
class FluidSolverInterface:
    def __init__(self, config):
        self.config = config
        self.flow_field_file = config['flow_field_file']
        self.time_dependent = config.get('flow_field_time_dependent', False)
        self.flow_field_data = None
        self.velocity_interpolator = None
        self.load_flow_field_data()

    def load_flow_field_data(self):
        """
        Load the flow field data from the specified file.
        """
        try:
            with h5py.File(self.flow_field_file, 'r') as f:
                # Assuming the HDF5 file contains datasets 'x', 'y', 'z', 'u', 'v', 'w'
                self.x = f['x'][:]
                self.y = f['y'][:]
                self.z = f['z'][:]
                self.u = f['u'][:]
                self.v = f['v'][:]
                self.w = f['w'][:]

                if self.time_dependent:
                    # Handle time-dependent data
                    self.times = f['times'][:]
                    # u, v, w should be 4D arrays with time as the first dimension
                else:
                    # u, v, w are 3D arrays
                    self.create_interpolator()
        except Exception as e:
            raise IOError(f"Error loading flow field data: {e}")

    def create_interpolator(self):
        """
        Create interpolator functions for u, v, w components.
        """
        self.u_interp = RegularGridInterpolator(
            (self.x, self.y, self.z), self.u, bounds_error=False, fill_value=None
        )
        self.v_interp = RegularGridInterpolator(
            (self.x, self.y, self.z), self.v, bounds_error=False, fill_value=None
        )
        self.w_interp = RegularGridInterpolator(
            (self.x, self.y, self.z), self.w, bounds_error=False, fill_value=None
        )

    def update_flow_field(self, current_time):
        """
        Update the flow field data for the current simulation time.
        """
        if self.time_dependent:
            # Interpolate the flow field data in time
            # Find the indices surrounding the current time
            time_indices = np.searchsorted(self.times, current_time, side='right')
            if time_indices == 0 or time_indices == len(self.times):
                raise ValueError("Current time is out of bounds of the flow field data.")

            t0 = self.times[time_indices - 1]
            t1 = self.times[time_indices]
            weight = (current_time - t0) / (t1 - t0)

            # Interpolate u, v, w between t0 and t1
            u_t0 = self.u[time_indices - 1]
            u_t1 = self.u[time_indices]
            self.u_current = (1 - weight) * u_t0 + weight * u_t1

            v_t0 = self.v[time_indices - 1]
            v_t1 = self.v[time_indices]
            self.v_current = (1 - weight) * v_t0 + weight * v_t1

            w_t0 = self.w[time_indices - 1]
            w_t1 = self.w[time_indices]
            self.w_current = (1 - weight) * w_t0 + weight * w_t1

            # Update interpolators with current data
            self.create_interpolator_time_dependent()
        # For time-independent flow fields, no update is necessary

    def create_interpolator_time_dependent(self):
        """
        Create interpolator functions for the current time step in time-dependent data.
        """
        self.u_interp = RegularGridInterpolator(
            (self.x, self.y, self.z), self.u_current, bounds_error=False, fill_value=None
        )
        self.v_interp = RegularGridInterpolator(
            (self.x, self.y, self.z), self.v_current, bounds_error=False, fill_value=None
        )
        self.w_interp = RegularGridInterpolator(
            (self.x, self.y, self.z), self.w_current, bounds_error=False, fill_value=None
        )

    def get_velocity_at(self, position):
        """
        Return the interpolated velocity at the given position.
        :param position: numpy array of shape (3,)
        :return: numpy array of shape (3,) containing velocity components (u, v, w)
        """
        u = self.u_interp(position).item()
        v = self.v_interp(position).item()
        w = self.w_interp(position).item()
        return np.array([u, v, w])
