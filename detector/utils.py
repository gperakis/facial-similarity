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


def plot_triplet_images(ref_image_path,
                        sim_image_path,
                        dif_image_path,
                        format='pgm'):
    """

    :param ref_image_path:
    :param sim_image_path:
    :param dif_image_path:
    :return:
    """
    ref_image = plt.imread(ref_image_path, format=format)
    sim_image = plt.imread(sim_image_path, format=format)
    dif_image = plt.imread(dif_image_path, format=format)

    draw_image(131,
               ref_image,
               "Reference")

    draw_image(132,
               sim_image,
               "Similar")

    draw_image(133,
               dif_image,
               "Different")

    plt.tight_layout()
    plt.show()


def plot_random_image_transformations(data_generator,
                                      img_path):
    sid = 150
    np.random.seed(42)

    image = plt.imread(img_path, format='jpg')

    sid += 1

    draw_image(sid, image, "orig")

    for j in range(4):
        augmented = data_generator.random_transform(image)
        sid += 1
        draw_image(sid, augmented, "aug#{:d}".format(j + 1))

    plt.tight_layout()
    plt.show()


def plot_history(history):
    """

    :param history:
    :return:
    """

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
