import h5py
import numpy as np

# Define the grid and a simple velocity field
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
z = np.linspace(0, 1, 10)

# Example: Uniform flow in x-direction, zero in y and z
u = np.ones((10, 10, 10))  # Velocity in x-direction
v = np.zeros((10, 10, 10))  # Velocity in y-direction
w = np.zeros((10, 10, 10))  # Velocity in z-direction

# Save to HDF5
with h5py.File('flow_field_data.h5', 'w') as f:
    f.create_dataset('x', data=x)
    f.create_dataset('y', data=y)
    f.create_dataset('z', data=z)
    f.create_dataset('u', data=u)
    f.create_dataset('v', data=v)
    f.create_dataset('w', data=w)

print("Mock flow field file 'flow_field_data.h5' created.")
