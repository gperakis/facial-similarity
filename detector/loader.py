"""
Custom Dataset Class
"""
from __future__ import division, print_function

import glob
import os
from itertools import combinations, product
from typing import List

import pandas as pd

from detector.config import Config

pd.set_option('display.expand_frame_repr', False)


class ImagesLoader:

    def __init__(self,
                 image_folder_dataset: str = Config.training_dir,
                 file_ext: str = "*.pgm"):
        """

        :param image_folder_dataset:
        :param file_ext:
        """
        self.image_folder_dataset = image_folder_dataset

        self.sub_dirs = sorted(os.listdir(self.image_folder_dataset))

        self.file_ext = file_ext

    def parse_filenames(self):
        """

        :return:
        """
        filenames = list()
        # for each subdirectory we have a number of .pgm files that we want to load
        for sub_dir in self.sub_dirs:

            # creating the actual path of the sub-directory
            path = os.path.join(self.image_folder_dataset,
                                sub_dir,
                                self.file_ext)

            # for each filename, obtain the target (from the filename's description)
            # and the actual time series.
            for fn in glob.glob(path):
                filenames.append(fn)

        return filenames
