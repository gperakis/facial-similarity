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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param cm:
    :param classes:
    :param normalize:
    :param cmap:
    :return:
    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'

    else:
        title = 'Confusion matrix'

    plt.imshow(cm,
               interpolation='nearest',
               cmap=cmap)

    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def full_multi_class_report(model,
                            x,
                            y_true,
                            classes,
                            batch_size=32,
                            binary=False):
    """
    Μulti-class or binary report
    If binary (sigmoid output), set binary parameter to True

    :param model:
    :param x:
    :param y_true:
    :param classes:
    :param batch_size:
    :param binary:
    :return:
    """
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true, axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.predict_classes(x, batch_size=batch_size)

    # 3. Print accuracy score
    print("Accuracy: {:.3f} %".format(100 * accuracy_score(y_true, y_pred)))

    print("")

    # 4. Print classification report
    print("Classification Report", end='\n\n')

    print(classification_report(y_true,
                                y_pred,
                                digits=5))

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(cnf_matrix,
                          classes=classes)


if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data
    y = to_categorical(iris.target)
    labels_names = iris.target_names

    xid, yid = 0, 1

    le = LabelEncoder()
    encoded_labels = le.fit_transform(iris.target_names)

    plt.scatter(x[:, xid], x[:, yid], c=y, cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(iris.feature_names[xid])
    plt.ylabel(iris.feature_names[yid])
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        train_size=0.8,
                                                        random_state=seed)

    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      train_size=0.8,
                                                      random_state=seed)

    # Basic Keras Model
    # Create a very basic MLNN with a single Dense layer.

    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(4,)))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # hist = model.fit(x_train,
    #                  y_train,
    #                  epochs=200,
    #                  batch_size=16,
    #                  verbose=0,
    #                  validation_data=(x_val, y_val))
    #
    # plot_keras_history(hist)
    #
    # # Full report on the Validation Set
    # full_multi_class_report(model,
    #                         x_val,
    #                         y_val,
    #                         le.inverse_transform(np.arange(3)))
    #
    # # Full report on the test set
    # full_multi_class_report(model,
    #                         x_test,
    #                         y_test,
    #                         le.inverse_transform(np.arange(3)))

    # # Grid Search
    # # Using grid search in keras can lead to an issue when trying to use
    # # custom scoring with multiclass models.
    # # Assume you creates a multiclass model as above with Iris.
    # # With keras, you usually encode y as categorical data like this: [[0,1,0],[1,0,0], ...]
    # # But when you try to use a custom scoring such as below:
    #
    # y = iris.target
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8,
    #                                                   random_state=seed)
    #
    #
    # def create_model(dense_layers: List[int] = None,
    #                  activation='relu',
    #                  optimizer='rmsprop'):
    #     """
    #
    #     :param dense_layers:
    #     :param activation:
    #     :param optimizer:
    #     :return:
    #     """
    #     if dense_layers is None:
    #         dense_layers = [8]
    #
    #     model = Sequential()
    #
    #     for index, lsize in enumerate(dense_layers):
    #         # Input Layer - includes the input_shape
    #         if index == 0:
    #             model.add(Dense(lsize,
    #                             activation=activation,
    #                             input_shape=(4,)))
    #         else:
    #             model.add(Dense(lsize,
    #                             activation=activation))
    #
    #     model.add(Dense(3, activation='softmax'))
    #     model.compile(optimizer=optimizer,
    #                   loss='categorical_crossentropy',
    #                   metrics=['accuracy'])
    #     return model
    #
    #
    # model = KerasClassifier(build_fn=create_model,
    #                         epochs=10,
    #                         batch_size=5,
    #                         verbose=0)
    #
    # param_grid = {'dense_layers': [[4], [8], [8, 8]],
    #               'activation': ['relu', 'tanh'],
    #               'optimizer': ('rmsprop', 'adam'),
    #               'epochs': [10, 50],
    #               'batch_size': [5, 16]}
    #
    # grid = GridSearchCV(model,
    #                     param_grid=param_grid,
    #                     return_train_score=True,
    #                     scoring=['precision_macro', 'recall_macro', 'f1_macro'],
    #                     refit='precision_macro')
    #
    # grid_results = grid.fit(x_train, y_train)
    #
    # print('Parameters of the best model: ')
    # print(grid_results.best_params_)
