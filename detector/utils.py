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
