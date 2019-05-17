#!/bin/env/python3
# -*- encoding: utf-8 -*-

"""
Configuration Class
A simple class to manage configuration
"""
import os

from detector import DATA_DIR


class Config:
    """

    """

    training_dir = os.path.join(DATA_DIR, 'faces', 'training_keras')
    testing_dir = os.path.join(DATA_DIR, 'faces', 'testing_keras')

    train_batch_size = 8
    train_number_epochs = 500

    img_width = 150
    img_height = 150
