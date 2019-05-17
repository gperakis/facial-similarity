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
from typing import Tuple
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
                 val_size=0.3):
        """

        :param data:
        :param val_size:
        """
        self.data = data
        self.val_size = val_size

    def train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function splits the dataset in train and validation sets.
        :return:
        """
        data_len = len(self.data)

        train_size = np.floor((1 - self.val_size) * data_len)

        train_data = self.data.loc[:train_size - 1]
        val_data = self.data.loc[train_size:]

        return train_data, val_data

    def augment_images2(self, fnames, n_new: int = 6):
        """

        :param paths:
        :return:
        """
        # TO DO finish it
        datagen = ImageDataGenerator(**self.datagen_args)

        img_path = fnames[3]

        img = image.load_img(img_path, target_size=(Config.img_height,
                                                    Config.img_width))

        x = image.img_to_array(img)

        x = x.reshape((1,) + x.shape)

        plt.figure(0)
        imgplot = plt.imshow(image.array_to_img(x[0]))

        i = 1
        for batch in datagen.flow(x, batch_size=1):
            plt.figure(i)
            imgplot = plt.imshow(image.array_to_img(batch[0]))
            i += 1
            if i % 6 == 0:
                break

        plt.show()
