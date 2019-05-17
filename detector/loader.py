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

    def parse_filenames(self) -> List[str]:
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

    @staticmethod
    def is_different(paths) -> int:
        """
        Checking if the two images are from the same folder
         (a.k.a belong to a different person)
        :return:
        """
        folder1 = paths[0].split('/')[-2]
        folder2 = paths[1].split('/')[-2]

        return int(folder1 != folder2)

    def get_paths_with_targets(self,
                               fnames: List[str],
                               include_same_image: bool = True):
        """

        :param fnames:
        :param include_same_image:
        :return:
        """
        if include_same_image:
            combs = set([tuple(sorted(t)) for t in product(fnames, repeat=2)])
        else:
            # create all 2 images combinations
            combs = combinations(fnames, 2)

        pic_1_paths = list()
        pic_2_paths = list()

        targets = list()

        for paths_tuple in combs:
            is_dif = self.is_different(paths_tuple)

            pic_1_paths.append(paths_tuple[0])
            pic_2_paths.append(paths_tuple[1])

            targets.append(is_dif)

        data = pd.DataFrame({'image1': pic_1_paths,
                             'image2': pic_2_paths,
                             'target': targets})

        return data
