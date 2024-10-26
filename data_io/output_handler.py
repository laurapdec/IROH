import h5py
import numpy as np
import os

class OutputHandler:
    def __init__(self, config):
        # Open the HDF5 file with write mode
        self.output_file = h5py.File(config['output_file'], 'w')
        self.export_interval = config.get('export_interval', 1)
        self.last_export_time = 0.0
        self.export_directory = config.get('export_directory', 'exported_data')
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)

    def save_state(self, time, particles):
        # Generate a unique group name
        group_name = f"time_{time:.2f}"
        count = 0
        while group_name in self.output_file:
            count += 1
            group_name = f"time_{time:.2f}_{count}"
        
        # Create a new group with the unique name
        group = self.output_file.create_group(group_name)
        positions = np.array([particle.position for particle in particles])
        properties = [particle.properties for particle in particles]
        
        # Save scalar properties
        scalar_names = properties[0].keys()
        for name in scalar_names:
            data = np.array([prop[name] for prop in properties])
            group.create_dataset(name, data=data)
        
        # Save particle positions
        group.create_dataset('positions', data=positions)

        # Export data to .dat files if export interval is reached
        if time - self.last_export_time >= self.export_interval:
            self.export_dat_files(time, positions, properties)
            self.last_export_time = time

    def export_dat_files(self, time, positions, properties):
        # Format time for file name
        time_str = f"{time:.2f}"
        scalar_names = properties[0].keys()
        data = positions
        headers = ['x', 'y', 'z']
        
        # Append scalar properties to data for export
        for name in scalar_names:
            scalar_data = np.array([prop[name] for prop in properties]).reshape(-1, 1)
            data = np.hstack((data, scalar_data))
            headers.append(name)
        
        # Save .dat file with appropriate headers
        data_file = os.path.join(self.export_directory, f"data_{time_str}.dat")
        header_line = ' '.join(headers)
        np.savetxt(data_file, data, header=header_line, comments='')

    def save_scalar_variance(self, time, variance):
        scalar_variance_file = os.path.join(self.export_directory, 'scalar_variance.dat')
        
        # Write headers if the file doesn't already exist
        if not os.path.exists(scalar_variance_file):
            with open(scalar_variance_file, 'w') as f:
                f.write('time variance\n')
        
        # Append variance data to the file
        with open(scalar_variance_file, 'a') as f:
            f.write(f"{time:.6f} {variance:.6f}\n")
