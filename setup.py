#!/usr/bin/env python3
"""
Bull Machine v1.7.2 Setup Configuration
Production-ready trading engine package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Advanced multi-domain confluence trading system"

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements-production.txt"
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, "r") as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith("#")
        ]
else:
    # Fallback requirements
    requirements = [
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "quantlib>=1.31",
        "TA-Lib>=0.4.25",
        "pyyaml>=6.0",
        "structlog>=23.1.0",
        "python-dotenv>=1.0.0",
        "numba>=0.57.0"
    ]

setup(
    name="bull-machine",
    version="1.7.2",
    author="Bull Machine Team",
    author_email="team@bullmachine.ai",
    description="Advanced multi-domain confluence trading system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bullmachine/bull-machine",

    # Package discovery
    packages=find_packages(exclude=["tests*", "scripts*", "docs*", "results*"]),

    # Include package data
    package_data={
        "bull_machine": [
            "configs/**/*.json",
            "configs/**/*.yaml",
            "data/**/*.csv",
        ]
    },
    include_package_data=True,

    # Dependencies
    install_requires=requirements,

    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "mypy>=1.5.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
        "db": [
            "sqlalchemy>=2.0.0",
            "alembic>=1.11.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
        ]
    },

    # Console scripts for CLI tools
    entry_points={
        "console_scripts": [
            "bull-machine=run_full_confluence_backtest:main",
            "bull-backtest=bin.production_backtest:main",
            "bull-analyze=scripts.research.analyze_btc_with_bull_machine:main",
            "bull-confluence=run_full_confluence_backtest:main",
        ]
    },

    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],

    # Python version requirement
    python_requires=">=3.8",

    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/bullmachine/bull-machine/issues",
        "Source": "https://github.com/bullmachine/bull-machine",
        "Documentation": "https://docs.bullmachine.ai",
    },

    # Zip safety
    zip_safe=False,

    # Keywords for discovery
    keywords="trading, algorithmic-trading, confluence, technical-analysis, quantitative-finance",
)
