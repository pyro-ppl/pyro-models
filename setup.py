# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup
from codecs import open
from os import path

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='Pyro Models',
    version='0.0.0',
    description='Models written in Pyro Probabilistic Programming Language',
    long_description=readme(),
    packages=find_packages(exclude=['contrib', 'docs', 'tests*', 'examples']),
    url='https://github.com/pyro-ppl/pyro-models',
    author='Uber AI Labs',
    author_email='jpchen@uber.com',
    install_requires=[
        'pyro-ppl>=0.3',
        'torch>=1.0.0',
    ],
    extras_require={
        'test': ['flake8', 'pytest>=4.1'],
    },
    tests_require=['flake8', 'pytest>=4.1'],
    keywords='probabilistic machine learning bayesian statistics pytorch',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
)
