#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="nn_optimizer",
    version="0.1",
    description="NN active learning for molecular geometry optimization.",
    url="https://github.com/yilinyang1/nn_optimizer",
    author="Yilin Yang",
    author_email="yiliny2@andrew.cmu.edu",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["ase"],
    long_description="""NN active learning for molecular geometry optimization.""",
)