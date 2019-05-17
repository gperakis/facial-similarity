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


def predict_similarity_from_filename(model: Model,
                                     img1_path,
                                     img2_path,
                                     dissimilarity_threshold: float = 0.65):
    """

    :param model:
    :param img1_path:
    :param img2_path:
    :param dissimilarity_threshold:
    :return:
    """
    # loading first image
    img1 = image.load_img(img1_path,
                          target_size=(Config.img_height,
                                       Config.img_width))

    # loading second image
    img2 = image.load_img(img2_path,
                          target_size=(Config.img_height,
                                       Config.img_width))

    # converting first image to array
    img1_array = image.img_to_array(img1)
    # normalizing first image
    img1_array_norm = img1_array / 255.

    # converting second image to array
    img2_array = image.img_to_array(img2)
    # normalizing second image
    img2_array_norm = img2_array / 255.

    # reshaping the first image from 3D to 4D tensor in order to be able to pass
    # it through the trained model.
    # (1 sample, width, height, # channels)
    img1_array_norm = img1_array_norm.reshape((1,) + img1_array_norm.shape)

    # reshaping the first image from 3D to 4D tensor in order to be able to pass
    # it through the trained model.
    # (1 sample, width, height, # channels)
    img2_array_norm = img2_array_norm.reshape((1,) + img2_array_norm.shape)

    # Getting the prediction.
    # probability close to 1 for dissimilarity
    pred = model.predict(x={'left_input': img1_array_norm,
                            'right_input': img2_array_norm})

    # plotting the two images side by side
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)
    plt.show()

    # Getting the actual probability of beiing dissimilar images.
    prob = pred[0][0]

    if prob > dissimilarity_threshold:
        print('Different people')
        return prob
    else:
        print('Same person')
        return prob


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


class SiameseNetworkModel(CustomModel):

    def __init__(self):
        """

        """
        super().__init__()

    @staticmethod
    def initialize_weights(shape, name=None):
        """
        The following paper: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf

        suggests to initialize CNN layer weights
        with mean as 0.0 and standard deviation of 0.01

        :param shape:
        :param name:
        :return:
        """

        return np.random.normal(loc=0.0,
                                scale=1e-2,
                                size=shape)

    @staticmethod
    def initialize_bias(shape, name=None):
        """
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01

        :param shape:
        :param name:
        :return:
        """

        return np.random.normal(loc=0.5,
                                scale=1e-2,
                                size=shape)

    def build_model_1(self,
                      img_dimension: int = Config.img_height):
        """

        :param img_dimension:
        :return:
        """

        input_shape = ((img_dimension ** 2) * 3,)

        convolution_shape = (img_dimension, img_dimension, 3)

        kernel_size_1 = (4, 4)

        kernel_size_2 = (3, 3)

        pool_size_1 = (3, 3)

        pool_size_2 = (2, 2)

        strides = 1

        sequential_conv_model = [

            Reshape(input_shape=input_shape,
                    target_shape=convolution_shape),

            Conv2D(32,
                   kernel_size=kernel_size_1,
                   strides=strides,
                   activation=None),

            LeakyReLU(alpha=0.2),

            Conv2D(32,
                   kernel_size=kernel_size_1,
                   strides=strides,
                   activation=None),

            LeakyReLU(alpha=0.2),

            MaxPooling2D(pool_size=pool_size_1,
                         strides=strides),

            Conv2D(64,
                   kernel_size=kernel_size_2,
                   strides=strides,
                   activation=None),

            LeakyReLU(alpha=0.2),

            Conv2D(64,
                   kernel_size=kernel_size_2,
                   strides=strides,
                   activation=None),

            LeakyReLU(alpha=0.2),

            MaxPooling2D(pool_size=pool_size_2,
                         strides=strides),

            Flatten(),

            Dense(64, activation=activations.sigmoid)

        ]

        seq_model = Sequential(sequential_conv_model)

        input_x1 = Input(shape=input_shape)
        input_x2 = Input(shape=input_shape)

        output_x1 = seq_model(input_x1)
        output_x2 = seq_model(input_x2)

        euclidean_distance = Lambda(
            lambda tensors: K.abs(tensors[0] - tensors[1]))([output_x1,
                                                             output_x2])

        outputs = Dense(1,
                        activation=activations.sigmoid)(euclidean_distance)

        model = models.Model([input_x1,
                              input_x2],
                             outputs)

        print(model.summary())

        self.model = model
        return self.model

    def build_model_2(self,
                      img_dimension: int = Config.img_height):
        """

        :param img_dimension:
        :return:
        """

        input_shape = (img_dimension, img_dimension, 3)

        # Define the tensors for the two input images
        left_input = Input(input_shape, name='left_input')
        right_input = Input(input_shape, name='right_input')

        # Convolutional Neural Network
        model = Sequential()

        model.add(Conv2D(64, (10, 10),
                         activation='relu',
                         input_shape=input_shape,
                         kernel_initializer=self.initialize_weights,
                         kernel_regularizer=l2(2e-4)))

        model.add(MaxPooling2D())

        model.add(Conv2D(128, (7, 7), activation='relu',
                         kernel_initializer=self.initialize_weights,
                         bias_initializer=self.initialize_bias,
                         kernel_regularizer=l2(2e-4)))

        model.add(MaxPooling2D())
        model.add(Conv2D(128, (4, 4),
                         activation='relu',
                         kernel_initializer=self.initialize_weights,
                         bias_initializer=self.initialize_bias,
                         kernel_regularizer=l2(2e-4)))

        model.add(MaxPooling2D())

        model.add(Conv2D(256, (4, 4),
                         activation='relu',
                         kernel_initializer=self.initialize_weights,
                         bias_initializer=self.initialize_bias,
                         kernel_regularizer=l2(2e-4)))

        model.add(Flatten())

        model.add(Dense(4096, activation='sigmoid',
                        kernel_regularizer=l2(1e-3),
                        kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias))

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))

        L1_distance = L1_layer([encoded_l, encoded_r])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense(1,
                           activation='sigmoid',
                           bias_initializer=self.initialize_bias,
                           name='main_output')(L1_distance)

        # Connect the inputs with the outputs
        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

        # return the model

        self.model = siamese_net

        print(model.summary())

        return siamese_net

    def build_model_3(self,
                      img_dimension: int = Config.img_height,
                      dr: float = 0.3):
        """

        :param img_dimension:
        :param dr:
        :return:
        """

        input_shape = (img_dimension, img_dimension, 3)

        # Define the tensors for the two input images
        left_input = Input(input_shape, name='left_input')
        right_input = Input(input_shape, name='right_input')

        # Convolutional Neural Network
        model = Sequential()

        model.add(Conv2D(64, (10, 10),
                         activation='relu',
                         input_shape=input_shape,
                         kernel_initializer=self.initialize_weights,
                         kernel_regularizer=l2(2e-4)))

        model.add(MaxPooling2D())

        model.add(Dropout(rate=dr))

        model.add(Conv2D(128, (7, 7), activation='relu',
                         kernel_initializer=self.initialize_weights,
                         bias_initializer=self.initialize_bias,
                         kernel_regularizer=l2(2e-4)))

        model.add(MaxPooling2D())

        model.add(Dropout(rate=dr))

        model.add(Conv2D(128, (4, 4),
                         activation='relu',
                         kernel_initializer=self.initialize_weights,
                         bias_initializer=self.initialize_bias,
                         kernel_regularizer=l2(2e-4)))

        model.add(MaxPooling2D())

        model.add(Dropout(rate=dr))

        model.add(Conv2D(256, (4, 4),
                         activation='relu',
                         kernel_initializer=self.initialize_weights,
                         bias_initializer=self.initialize_bias,
                         kernel_regularizer=l2(2e-4)))

        model.add(Flatten())

        model.add(Dense(4096, activation='sigmoid',
                        kernel_regularizer=l2(1e-3),
                        kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias))

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))

        L1_distance = L1_layer([encoded_l, encoded_r])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense(1,
                           activation='sigmoid',
                           bias_initializer=self.initialize_bias,
                           name='main_output')(L1_distance)

        # Connect the inputs with the outputs
        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

        # return the model

        self.model = siamese_net

        print(model.summary())

        return siamese_net

    def fit_model(self,
                  x_left,
                  x_right,
                  y,
                  x_val_left,
                  x_val_right,
                  y_val,
                  e: int = 30,
                  learning_rate: float = 0.00006,
                  add_callbacks: bool = True,
                  model_name: str = 'siamese_net.h5',
                  batch_size: int = Config.train_batch_size):
        """

        :param x_left:
        :param x_right:
        :param y:
        :param x_val_left:
        :param x_val_right:
        :param y_val:
        :param e:
        :param learning_rate:
        :param add_callbacks:
        :param model_name:
        :param batch_size:
        :return:
        """
        filepath = os.path.join(MODELS_DIR, model_name)

        self.model.compile(loss=losses.binary_crossentropy,
                           optimizer=optimizers.Adam(lr=learning_rate))

        train_inp = {'left_input': x_left,
                     'right_input': x_right}

        val_inp = {'left_input': x_val_left,
                   'right_input': x_val_right}

        callbacks = list()

        if add_callbacks:
            callbacks.extend(self._add_callbacks(filepath))

        history = self.model.fit(x=train_inp,
                                 y=y,
                                 epochs=e,
                                 validation_data=(val_inp, y_val),
                                 verbose=1,
                                 batch_size=batch_size)
        return history
