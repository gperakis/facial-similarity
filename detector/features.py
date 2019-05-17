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
                 val_size=0.2):
        """

        :param data:
        :param val_size:
        """
        self.data = data
        self.val_size = val_size

    def train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

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

    @staticmethod
    def get_augmented(datagen,
                      image_filename: str,
                      n_new_samples: int = 6) -> np.array:
        """
        This method, using a data_generator takes as input a image filename
        and does the following:
        1) Loads the images from directory
        2) Converts the image in a n-dimensional numpy array
        3) Normalizes the numpy array image by dividing with 255.0
        4) Creates a number of augmentd images by zooming, flipping, or tilding the image.
        5) Creates a numpy array of 4 dimensions (n_samples, height, width, # channels) from the
           original image and the augmented images.

        :param datagen:
        :param image_filename:
        :param n_new_samples:
        :return:
        """
        outputs = list()

        # loading image from directory
        img = image.load_img(image_filename,
                             target_size=(Config.img_height,
                                          Config.img_width))

        # converting image to numpy array
        x = image.img_to_array(img)

        # normalizing the source image.
        x_normalized = x / 255.

        # appending the source image to a list
        outputs.append(x_normalized)

        # reshaping the image into a 4d object (1, height, width, n-channels) in order
        # to pass it to keras's generator in order to produce more images
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1):

            # this is already normalized
            new_img_array = batch[0]
            i += 1

            # storing the augmented image to a list.
            outputs.append(new_img_array)

            if i % n_new_samples == 0:
                break

        # concatenate the original image with the K- new augmented images in a
        # 4D numpy array
        return np.array(outputs)
