import os
import pandas as pd
import numpy as np
import h5py

class LatexDataExporter:
    def __init__(self, export_directory="latex_input"):
        self.export_directory = export_directory
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)

    def export_data(self, filename, columns, data):
        """Exports continuous data with multiple rows to a specified .dat file."""
        df = pd.DataFrame(data, columns=columns).dropna().sort_values(by=columns[0])
        file_path = os.path.join(self.export_directory, filename)
        df.to_csv(file_path, index=False, sep="\t")

    def append_single_data_point(self, filename, label, data_point):
        """Appends a single data point to a .dat file with a label, without overwriting."""
        file_path = os.path.join(self.export_directory, filename)
        with open(file_path, 'a') as f:
            f.write(f"{label}\t{data_point}\n")

    # Individual export functions for each type of data
    def export_scalar_variance_decay(self, data):
        self.export_data("scalar_variance_decay_comparison.dat", ["Time", "Scalar Variance"], data)

    def export_mean_temperature_profiles(self, data):
        self.export_data("mean_temperature_profiles.dat", ["Axial Position", "Temperature"], data)

    def export_computational_times(self, data):
        self.export_data("computational_times.dat", ["Model", "Simulation Time (hours)", "Relative Computational Cost"], data)

    def export_rms_temperature_fluctuations(self, data):
        self.export_data("rms_temperature_fluctuations.dat", ["Radial Position", "RMS Temperature"], data)

    def export_mean_co_concentration(self, data):
        self.export_data("mean_co_concentration.dat", ["Position", "CO Concentration"], data)

    def export_simulation_time_vs_grid_resolution(self, data):
        self.export_data("simulation_time_vs_grid_resolution.dat", ["Grid Resolution", "Simulation Time"], data)

    def export_simulation_time_vs_particle_count(self, data):
        self.export_data("simulation_time_vs_particle_count.dat", ["Particle Count", "Simulation Time"], data)

    def export_temperature_contours(self, data):
        self.export_data("temperature_contours.dat", ["Axial Position", "Radial Position", "Temperature"], data)

    def export_key_findings_summary(self, data):
        self.export_data("key_findings_summary.dat", ["Metric", "Description"], data)

class OutputHandler:
    """Handles main output operations for simulation data and LaTeX exports."""
    
    def __init__(self, config):
        self.config = config
        self.output_file = h5py.File(config['output_file'], 'w')
        
        # Attributes and directory setup
        self.micromixing_model = config.get('micromixing_model', 'adaptive')
        self.output_file.attrs['micromixing_model'] = self.micromixing_model
        
        self.export_interval = config.get('export_interval', 1)
        self.last_export_time = 0.0
        self.export_directory = config.get('export_directory', 'exported_data')
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)
        
        # Initialize LaTeX data exporter
        self.latex_exporter = LatexDataExporter()

    def save_state(self, time, particles):
        """Saves particle positions and properties to HDF5 and triggers .dat exports if required."""
        group_name = f"time_{time:.2f}"
        count = 0
        while group_name in self.output_file:
            count += 1
            group_name = f"time_{time:.2f}_{count}"
        
        group = self.output_file.create_group(group_name)
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
        """Exports particle data to a .dat file for each time step as needed."""
        time_str = f"{time:.2f}"
        
        # Combine data and headers for export
        scalar_names = properties[0].keys()
        data = positions
        headers = ['x', 'y', 'z']
        
        # Append scalar properties to data for export
        for name in scalar_names:
            scalar_data = np.array([prop[name] for prop in properties]).reshape(-1, 1)
            data = np.hstack((data, scalar_data))
            headers.append(name)
        
        # Drop rows with NaN values, sort by first axis
        data = data[~np.isnan(data).any(axis=1)]
        data = data[data[:, 0].argsort()]

        # Save sorted and cleaned .dat file
        data_file = f"{self.micromixing_model}_data_{time_str}.dat"
        self.latex_exporter.export_data(data_file, headers, data)

    # LaTeX-specific exports
    def save_mean_temperature(self, time, mean_temp):
        self.latex_exporter.append_single_data_point("mean_temperature.dat", f"Time {time:.2f}", mean_temp)

    def save_rms_temperature(self, time, rms_temp):
        self.latex_exporter.append_single_data_point("rms_temperature.dat", f"Time {time:.2f}", rms_temp)

    def save_co_concentration(self, time, co_concentration):
        self.latex_exporter.append_single_data_point("co_concentration.dat", f"Time {time:.2f}", co_concentration)

    def save_scalar_variance(self, time, variance):
        self.latex_exporter.append_single_data_point("scalar_variance.dat", f"Time {time:.2f}", variance)
