from setuptools import find_packages, setup

setup(
    name='SpatialSort',
    version='0.1.0',
    description='A spatially aware Bayesian clustering approach that allows for the incorporation of prior biological knowledge.',
    author='Eric Lee',
    author_email='psylin223@gmail.com',
    url='https://github.com/Roth-Lab/SpatialSort',
    packages=find_packages(),
    license='MIT',
    entry_points={
        'console_scripts': [
            'SpatialSort=src.cli:main',
        ]
    }
)