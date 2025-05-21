import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:   
    requirements = f.read().splitlines()
    

setup(
    name="MLOPS-Project-1",
    version="0.1.0",
    author="kaan",
    packages=find_packages(),
    install_requires=requirements,
)