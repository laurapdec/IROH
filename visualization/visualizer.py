# visualization/visualizer.py

import matplotlib.pyplot as plt
import h5py
import numpy as np

class Visualizer:
    def __init__(self, data_file):
        self.data_file = h5py.File(data_file, 'r')

    def plot_scalar_field(self, time, scalar_name):
        time_group = self.data_file.get(f"time_{time:.2f}")
        positions = time_group['positions'][:]
        scalar_values = time_group[scalar_name][:]
        plt.scatter(positions[:, 0], positions[:, 1], c=scalar_values, cmap='jet', s=5)
        plt.colorbar(label=scalar_name)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f"{scalar_name} Distribution at Time {time:.2f}")
        plt.show()
