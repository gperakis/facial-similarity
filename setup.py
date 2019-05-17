#!/bin/env/python3
# -*- encoding: utf-8 -*-

from setuptools import setup
from setuptools.command.install import install as _install

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


class Install(_install):
    def run(self):
        _install.do_egg_install(self)


setup(
    cmdclass={'install': Install},
    name='facial-similarity',
    version='0.1.0',
    packages=['detector'],
    description='Detecting if a person is in a set of images',
    long_description=open('README.md', 'r').read(),
    author='George Perakis',
    author_email='perakisgeorgios[@]gmail.com',
    install_requires=requirements)
