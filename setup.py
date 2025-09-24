from setuptools import setup, find_packages

# Import version from single source of truth
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bull_machine'))
from version import __version__, get_version_banner

setup(
    name="bull_machine",
    version=__version__,
    description=f"{get_version_banner()} - Advanced Algorithmic Trading Engine with 7-Layer Confluence",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
    ],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["bull-machine=bull_machine.app.main:main"]},
)
