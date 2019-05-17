"""
Custom Dataset Class
"""
from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from detector.config import Config

pd.set_option('display.expand_frame_repr', False)


class ImageProcess:
    """

    """

    datagen_args = dict(rotation_range=30,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        rescale=1. / 255,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode='nearest')

    def __init__(self,
                 data: pd.DataFrame,
                 val_size=0.2):
        """

        :param data:
        :param val_size:
        """
        self.data = data
        self.val_size = val_size
