#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

ROOT_PACKAGE_NAME = 'HPTuner'

def parse_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name=ROOT_PACKAGE_NAME,
    version='0.0.1',
    author=['Egor Elchinov'],
    packages=['HPTuner'],
    long_description='hyperparameters tunung for the Markovitz investment algorithm',
    requirements=parse_requirements()
)


