# run_simulation.py

import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from core.engine import SimulationEngine

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    config = load_config('simulation_config.json')
    engine = SimulationEngine(config)
    engine.run()

if __name__ == "__main__":
    main()
