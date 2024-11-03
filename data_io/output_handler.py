import os
import pandas as pd
import numpy as np
import h5py
import time
import os
import pandas as pd
import numpy as np
import h5py

class LatexDataExporter:
    def __init__(self, simulation_label, export_directory="latex_input"):
        self.simulation_label = simulation_label
        self.export_directory = export_directory  # Directly use export_directory as a string
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)

    def export_data(self, filename, columns, data):
        """Exports continuous data with multiple rows to a specified .dat file."""
        df = pd.DataFrame(data, columns=columns).dropna().sort_values(by=columns[0])
        file_path = os.path.join(self.export_directory, filename)
        df.to_csv(file_path, index=False, sep="\t")

    def append_single_data_point(self, filename, label, data_point):
        """Appends or updates a single data point in a .dat file without headers."""
        file_path = os.path.join(self.export_directory, filename)

        # Load existing data if file exists
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep="\t", header=None)
            df.columns = ["Label", "Data Point"]
            
            # Replace row if label exists, else add a new row
            if self.simulation_label in df["Label"].values:
                df.loc[df["Label"] == self.simulation_label, "Data Point"] = data_point
            else:
                df = pd.concat([df, pd.DataFrame([[self.simulation_label, data_point]], columns=df.columns)], ignore_index=True)
        else:
            # Create new DataFrame if file doesn't exist
            df = pd.DataFrame([[self.simulation_label, data_point]], columns=["Label", "Data Point"])

        # Save file without headers
        df.to_csv(file_path, index=False, sep="\t", header=False)
    
    # Specific export functions for each data type
    def export_scalar_variance_decay(self, data):
        self.export_data("scalar_variance_decay_comparison.dat", ["Time", "Scalar Variance"], data)

    def export_mean_temperature_profiles(self, data):
        self.export_data("mean_temperature_profiles.dat", ["Axial Position", "Temperature"], data)

    # Simulation state export
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

        # Export additional data files
        self.export_dat_files(time, positions, properties)

    def export_dat_files(self, time, positions, properties):
        """Exports particle data to a .dat file for each time step as needed."""
        time_str = f"{time:.2f}"
        scalar_names = properties[0].keys()
        data = positions
        headers = ['x', 'y', 'z']
        
        # Append scalar properties to data for export
        for name in scalar_names:
            scalar_data = np.array([prop[name] for prop in properties]).reshape(-1, 1)
            data = np.hstack((data, scalar_data))
            headers.append(name)
        
        # Drop rows with NaN values, sort by the first axis
        data = data[~np.isnan(data).any(axis=1)]
        data = data[data[:, 0].argsort()]

        # Save sorted and cleaned .dat file
        data_file = f"{self.simulation_label}_data_{time_str}.dat"
        self.export_data(data_file, headers, data)

    # Specific export functions for data types
    def export_scalar_variance_decay(self, data):
        self.export_data("scalar_variance_decay_comparison.dat", ["Time", "Scalar Variance"], data)

    def export_mean_temperature_profiles(self, data):
        self.export_data("mean_temperature_profiles.dat", ["Axial Position", "Temperature"], data)

    def export_computational_time(self):
        """Appends or updates total computational time in the .dat file."""
        elapsed_time = time.time() - self.start_time
        label = f"{self.simulation_label}_Elapsed_Time"
        self.append_single_data_point("computational_times.dat", label, elapsed_time)

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

    def close(self):
        """Closes the HDF5 file to finalize output."""
        self.output_file.close()
