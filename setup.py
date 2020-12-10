from setuptools import setup, find_packages

setup(
    name='SDSS_PAE',
    version='0.1.0',
    url='https://github.com/AlexGKim/SDSS_PAE',
    author='Alex Kim, Vanessa Boehm',
    description='PAE code for spectral data',
    packages=find_packages(),
    install_requires=['tensorflow_datasets', 'tensorflow-gpu', 'astropy', 'numpy'],
)
