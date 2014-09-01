#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# [The "New BSD" license]
# Copyright (c) 2014 The Board of Trustees of The University of Alabama
# All rights reserved.
#
# See LICENSE for details.

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirementstxt = f.read().splitlines()

setup(
    name='Changeset Feature Location',
    version='0.0.1',
    description='',
    long_description=readme,
    author='Christopher S. Corley',
    author_email='cscorley@crimson.ua.edu',
    url='https://github.com/cscorley/feature-location',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'data', 'paper', 'lib')),
    keywords = [],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Version Control",
        "Topic :: Text Processing",
        ],
    py_modules=['cfl'],
    install_requires=requirementstxt,
    entry_points='''
        [console_scripts]
        cfl=src:main
    ''',
)
