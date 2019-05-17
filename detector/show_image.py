#!/bin/env/python3
# -*- encoding: utf-8 -*-

"""
Module for plotting images.
"""
from __future__ import division, print_function

import os

from keras.preprocessing.image import ImageDataGenerator

from detector.config import Config
from detector.utils import plot_triplet_images, plot_random_image_transformations

if __name__ == "__main__":
    # Plotting Triplet Images
    DATA_DIR = Config.training_dir

    IMAGE_DIR = os.path.join(DATA_DIR, "s1")

    ref_image = os.path.join(IMAGE_DIR, "1.pgm")
    sim_image = os.path.join(IMAGE_DIR, "2.pgm")
    dif_image = os.path.join(IMAGE_DIR, "3.pgm")

    plot_triplet_images(ref_image, sim_image, dif_image)

    # Plotting Random Images Transformations

    datagen_args = dict(rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.4,
                        zoom_range=0.3,
                        horizontal_flip=True)

    datagen = ImageDataGenerator(**datagen_args)

    DATA_DIR = Config.training_dir

    IMAGE_DIR = os.path.join(DATA_DIR, "p1")

    image_path = os.path.join(IMAGE_DIR, "FT18_WhoIsWho_html_1b746456c526d0e9.jpg")

    plot_random_image_transformations(datagen, image_path)
