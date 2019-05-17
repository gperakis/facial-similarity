import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from typing import List

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

seed = 1000


def plot_keras_history(history):
    """

    :param history:
    :return:
    """

    acc = history.history.get('acc', [])

    val_acc = history.history.get('val_acc', [])

    loss = history.history.get('loss', [])

    val_loss = history.history.get('val_loss', [])

    if len(loss) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(loss) + 1)

    model_loss_text = "Training loss: {:.5f}".format(loss[-1])

    # Loss
    plt.figure(1)
    plt.plot(epochs, loss, 'b', label=model_loss_text)

    if val_loss:
        model_val_loss_text = "Validation loss: {:.5f}".format(val_loss[-1])

        plt.plot(epochs, val_loss, 'g', label=model_val_loss_text)

    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.figure(2)

    model_acc_text = "Training accuracy: {:.5f}".format(100 * acc[-1])

    plt.plot(epochs, acc, 'b', label=model_acc_text)

    if val_acc:
        val_acc_text = "Training accuracy: {:.5f}".format(100 * val_acc[-1])

        plt.plot(epochs, val_acc, 'g', label=val_acc_text)

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
