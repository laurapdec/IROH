# data_io/output_handler.py

import h5py
import numpy as np
import os

class OutputHandler:
    def __init__(self, config):
        self.output_file = h5py.File(config['output_file'], 'w')
        self.export_interval = config.get('export_interval', 1)
        self.last_export_time = 0.0
        self.export_directory = config.get('export_directory', 'exported_data')
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)

    def save_state(self, time, particles):
        group = self.output_file.create_group(f"time_{time:.2f}")
        positions = np.array([particle.position for particle in particles])
        properties = [particle.properties for particle in particles]
        scalar_names = properties[0].keys()
        for name in scalar_names:
            data = np.array([prop[name] for prop in properties])
            group.create_dataset(name, data=data)
        group.create_dataset('positions', data=positions)

        if time - self.last_export_time >= self.export_interval:
            self.export_dat_files(time, positions, properties)
            self.last_export_time = time

    def export_dat_files(self, time, positions, properties):
        time_str = f"{time:.2f}"
        # Export positions and scalar properties together
        scalar_names = properties[0].keys()
        data = positions
        headers = ['x', 'y', 'z']
        for name in scalar_names:
            scalar_data = np.array([prop[name] for prop in properties]).reshape(-1, 1)
            data = np.hstack((data, scalar_data))
            headers.append(name)
        data_file = os.path.join(self.export_directory, f"data_{time_str}.dat")
        header_line = ' '.join(headers)
        np.savetxt(data_file, data, header=header_line, comments='')
        # Export scalar variance data if available
        # (Assuming scalar variance data is stored separately)

    def save_scalar_variance(self, time, variance):
        scalar_variance_file = os.path.join(self.export_directory, 'scalar_variance.dat')
        if not os.path.exists(scalar_variance_file):
            with open(scalar_variance_file, 'w') as f:
                f.write('time variance\n')
        with open(scalar_variance_file, 'a') as f:
            f.write(f"{time:.6f} {variance:.6f}\n")
