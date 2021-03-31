from pathlib import Path

from setuptools import setup, find_packages


name = 'FBPAug'

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

setup(
    name=name,
    packages=find_packages(include=(name,)),
    descriprion='A collection of tools for deep learning experiments',
    install_requires=requirements,
    # OPTIONAL: uncomment if needed
    # python_requires='>=3.6',
)
