import tensorflow_datasets as tfds
import numpy as np


def load_mnist():

    train = tfds.load("mnist", split="train", as_supervised=True)
    test = tfds.load("mnist", split="test", as_supervised=True)

    def convert(ds):
        images = []
        labels = []

        for img, label in tfds.as_numpy(ds):
            img = img / 255.0
            img = img.reshape(28,28,1)
            images.append(img)
            labels.append(label)

        return np.array(images), np.array(labels)

    return convert(train), convert(test)


def load_cifar10():

    train = tfds.load("cifar10", split="train", as_supervised=True)
    test = tfds.load("cifar10", split="test", as_supervised=True)

    x_train = []
    y_train = []

    for image, label in tfds.as_numpy(train):
        x_train.append(image)
        y_train.append(label)

    x_test = []
    y_test = []

    for image, label in tfds.as_numpy(test):
        x_test.append(image)
        y_test.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)