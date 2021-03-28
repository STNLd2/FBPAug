from pathlib import Path

from setuptools import setup, find_packages

# automatically detect the lib name
dirs = {d.parent.name for d in Path(__file__).resolve().parent.glob('*/__init__.py') if d.parent.is_dir()}
dirs -= {'configs', 'notebooks', 'scripts', 'tests'}
dirs = [n for n in dirs if not n.startswith(('_', '.'))]
dirs = [n for n in dirs if not n.endswith(('.egg-info',))]

assert len(dirs) == 1, dirs
name = dirs[0]
if name == 'my-project-name':
    raise ValueError('Rename "my-project-name" to your project\'s name.')

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
