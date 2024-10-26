# data_io/input_handler.py

import json
import h5py
import os

class InputHandler:
    def __init__(self, config):
        self.config = config
        self.validate_config()
        self.flow_field_data = None
        self.chemical_mechanism = None

    def validate_config(self):
        """
        Validate the configuration parameters.
        """
        required_keys = [
            'time_step',
            'total_time',
            'output_file',
            'initial_conditions',
            'flow_field_file',
            'mechanism_file'
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration parameter: '{key}' in configuration.")

        # Additional validation can be added here (e.g., check data types)

    def load_flow_field_data(self):
        """
        Load the flow field data from the specified file.
        """
        flow_field_file = self.config['flow_field_file']
        if not os.path.exists(flow_field_file):
            raise FileNotFoundError(f"Flow field file '{flow_field_file}' not found.")
        try:
            self.flow_field_data = h5py.File(flow_field_file, 'r')
        except Exception as e:
            raise IOError(f"Error loading flow field data from '{flow_field_file}': {e}")

    def get_flow_field_data(self):
        """
        Get the flow field data, loading it if necessary.
        """
        if self.flow_field_data is None:
            self.load_flow_field_data()
        return self.flow_field_data

    def load_chemical_mechanism(self):
        """
        Load the chemical mechanism from the specified file.
        """
        mechanism_file = self.config['mechanism_file']
        if not os.path.exists(mechanism_file):
            raise FileNotFoundError(f"Mechanism file '{mechanism_file}' not found.")
        try:
            with open(mechanism_file, 'r') as f:
                # Placeholder: Load and parse the mechanism file as needed
                self.chemical_mechanism = f.read()
        except Exception as e:
            raise IOError(f"Error loading chemical mechanism from '{mechanism_file}': {e}")

    def get_chemical_mechanism(self):
        """
        Get the chemical mechanism data, loading it if necessary.
        """
        if self.chemical_mechanism is None:
            self.load_chemical_mechanism()
        return self.chemical_mechanism

    def get_initial_conditions(self):
        """
        Return the initial conditions for particles.
        """
        return self.config['initial_conditions']

    def get_time_step(self):
        """
        Return the simulation time step.
        """
        return self.config['time_step']

    def get_total_time(self):
        """
        Return the total simulation time.
        """
        return self.config['total_time']

    def get_output_file(self):
        """
        Return the output file path.
        """
        return self.config['output_file']

    def get_export_interval(self):
        """
        Return the data export interval.
        """
        return self.config.get('export_interval', 1)

    def get_export_directory(self):
        """
        Return the export directory for .dat files.
        """
        return self.config.get('export_directory', 'exported_data')

    def get_micromixing_constant(self):
        """
        Return the micromixing constant.
        """
        return self.config.get('micromixing_constant', 1.0)

    def get_diffusivity(self):
        """
        Return the diffusivity value.
        """
        return self.config.get('diffusivity', 1e-5)

    def get_flow_field_time_dependent(self):
        """
        Return whether the flow field is time-dependent.
        """
        return self.config.get('flow_field_time_dependent', False)

    def get_config(self):
        """
        Return the full configuration dictionary.
        """
        return self.config
