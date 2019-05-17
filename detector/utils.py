import matplotlib.pyplot as plt
import numpy as np


def show_image(image: np.array,
               text: str = None,
               should_save: bool = False):
    """

    :param image:
    :param text:
    :param should_save:
    :return:
    """
    numpy_image = image.numpy()

    plt.axis("off")

    if text:
        plt.text(75,
                 8,
                 text,
                 style='italic',
                 fontweight='bold',
                 bbox={'facecolor': 'white',
                       'alpha': 0.8,
                       'pad': 10})

    plt.imshow(np.transpose(numpy_image, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    """

    :param iteration:
    :param loss:
    :return:
    """
    plt.plot(iteration, loss)
    plt.show()


def draw_image(subplot,
               image,
               title):
    """

    :param subplot:
    :param image:
    :param title:
    :return:
    """
    plt.subplot(subplot)

    plt.imshow(image)

    plt.title(title)
    plt.xticks([])
    plt.yticks([])
