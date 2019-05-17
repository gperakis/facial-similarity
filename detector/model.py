import os

import keras.backend as K
import numpy as np
from keras import models, activations, losses, optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense, Input, Lambda, LeakyReLU, \
    Dropout
from keras.models import Model
from keras.models import Sequential, load_model, save_model
from keras.preprocessing import image
from keras.regularizers import l2

from detector import TENSORBOARD_LOGS_DIR, MODELS_DIR
from detector.config import Config
import matplotlib.pyplot as plt


class CustomModel:
    def __init__(self):
        """

        """
        self.model: Model = None

    def _save_model(self, filepath: str):
        """
        This method saves the trained model to the specified filepath.
        :return:
        """
        save_model(self.model,
                   filepath=filepath)

    def load_model(self, filepath: str):
        """
        This method loads a trained model from a specified filename.
        It will search for the model in the MODELS directory.

        :return:
        """
        filepath = os.path.join(MODELS_DIR,
                                filepath)

        # checking if the model is already at scope
        if self.model is None:
            self.model = load_model(filepath=filepath)

        return self.model

    @staticmethod
    def _add_callbacks(model_full_path) -> list:
        """
        This method gives as the utility to add pre-determined callbacks to the
        callbacks list when trying to train our models.

        These callbacks consist of:
        The Tensorboard callback
        An Early Stopping callback
        A model checkpoint callback tha saves the model when the parameters are fullfilled.
        A callback tha reduces the learing rate of the model whenever reaches a plateau

        :return: A list of callbacks that we want to pass to the model fit.
        """
        monitor = 'val_loss'

        callbacks = [

            TensorBoard(log_dir=TENSORBOARD_LOGS_DIR,
                        histogram_freq=0,
                        embeddings_freq=0,
                        write_graph=True,
                        write_images=False),

            EarlyStopping(monitor=monitor,
                          patience=6,
                          verbose=1),

            ModelCheckpoint(filepath=model_full_path,
                            monitor=monitor,
                            save_best_only=True,
                            verbose=1),

            ReduceLROnPlateau(monitor=monitor,
                              factor=0.1,
                              patience=5,
                              verbose=1)]

        return callbacks
