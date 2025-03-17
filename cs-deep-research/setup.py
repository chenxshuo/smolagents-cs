# -*- coding: utf-8 -*-
""""""

import logging

from setuptools import find_packages, setup


logger = logging.getLogger(__name__)

setup(
    name="cs_deep_research",  # Replace with your package name
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # Looks for packages under src/
    install_requires=[],  # Add dependencies if needed
    author="Shuo Chen",
    author_email="chenshuo.cs@outlook.com",
    description="A deep-research agent based on smolagents",
    url="https://github.com/chenxshuo/smolagents-cs",  # Replace with your repo URL if applicable
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
