# setup.py

from setuptools import setup, find_packages

setup(
    name='iroh',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'h5py',
        'matplotlib',
        'scipy'
        # Add other dependencies
    ],
    entry_points={
        'console_scripts': [
            'run_iroh = run_simulation:main',
        ],
    },
)
