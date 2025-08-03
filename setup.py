from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="autoinsights",
    version="1.0.0",
    author="AutoInsights Team",
    author_email="contact@autoinsights.com",
    description="A comprehensive Business Intelligence platform for automated data analysis, churn prediction, and revenue forecasting",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/autoinsights",
    packages=find_packages(),
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "autoinsights=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="machine-learning, business-intelligence, data-science, streamlit, flask, mysql",
    project_urls={
        "Bug Reports": "https://github.com/your-username/autoinsights/issues",
        "Source": "https://github.com/your-username/autoinsights",
        "Documentation": "https://github.com/your-username/autoinsights#readme",
    },
) 