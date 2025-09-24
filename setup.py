from setuptools import setup, find_packages

setup(
    name="bull_machine",
    version="1.4.2",
    description="Bull Machine v1.4.2 - Advanced Algorithmic Trading Engine with 7-Layer Confluence",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
    ],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["bull-machine=bull_machine.app.main:main"]},
)
