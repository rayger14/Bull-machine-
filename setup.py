from setuptools import setup, find_packages

setup(
    name="bull_machine",
    version="1.1",
    description="Bull Machine v1.1 - Algorithmic Trading Engine",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
    ],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["bull-machine=bull_machine.app.main:main"]},
)
