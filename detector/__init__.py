#!/usr/bin/env python3
"""
Transforms app folder to python package
"""
import os
from os import makedirs
from os.path import dirname, abspath, exists

PARENT_DIRECTORY = dirname(dirname(abspath(__file__)))

DATA_DIR = os.path.join(PARENT_DIRECTORY, 'data')
MODELS_DIR = os.path.join(PARENT_DIRECTORY, 'models')

TENSORBOARD_LOGS_DIR = os.path.join(PARENT_DIRECTORY, 'Graph')

# if the folders don't exist, create them.
if not exists(DATA_DIR):
    makedirs(DATA_DIR)

if not exists(MODELS_DIR):
    makedirs(MODELS_DIR)

if not exists(TENSORBOARD_LOGS_DIR):
    makedirs(TENSORBOARD_LOGS_DIR)

# if the folders don't exist, create them.
if not exists(DATA_DIR):
    makedirs(DATA_DIR)

if not exists(MODELS_DIR):
    makedirs(MODELS_DIR)
