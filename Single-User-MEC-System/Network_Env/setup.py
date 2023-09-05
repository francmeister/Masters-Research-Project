from setuptools import setup, find_packages

setup(name='Network_Env',
      version='0.0.1',
      packages=find_packages() + find_packages(exclude=['Resources']),
      install_requires=['gym'])