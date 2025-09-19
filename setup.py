#!/usr/bin/env python3
"""
setup.py
Purpose: Package installation and metadata configuration for Akia HRM
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Akia HRM - Hierarchical Reasoning Model"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle optional dependencies
                    if not line.startswith('# '):
                        requirements.append(line.split('#')[0].strip())
    return requirements

setup(
    name="akia-hrm",
    version="1.4.0",
    author="Akia Team",
    author_email="team@akia.ai",
    description="Akia Hierarchical Reasoning Model - 27M Parameter Efficient AI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/coddey-me/akia-h1.4",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "flash": [
            "flash-attn>=2.0.0",
        ],
        "apex": [
            "apex",
        ],
        "advanced": [
            "deepspeed>=0.9.0",
            "lion-pytorch>=0.0.6",
        ],
        "all": [
            "flash-attn>=2.0.0",
            "apex",
            "deepspeed>=0.9.0",
            "lion-pytorch>=0.0.6",
        ]
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "akia-train=scripts.train_kaggle:main",
            "akia-evaluate=scripts.evaluate_model:main",
            "akia-export=scripts.export_model:main",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    keywords="artificial-intelligence, machine-learning, natural-language-processing, pytorch, transformer, reasoning, hierarchical-model",
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/coddey-me/akia-h1.4/issues",
        "Source": "https://github.com/coddey-me/akia-h1.4",
        "Documentation": "https://github.com/coddey-me/akia-h1.4/blob/main/docs/",
        "Kaggle Training": "https://github.com/coddey-me/akia-h1.4/blob/main/notebooks/akia_kaggle_training.ipynb",
    },
)
