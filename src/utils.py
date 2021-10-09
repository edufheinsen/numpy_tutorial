import gzip
import itertools
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads in MNIST handwritten digits dataset from "data" directory.

    Returns:
        X_train: (60000, 784) array containing training image pixel data
        X_test: (10000, 784) array containing testing image pixel data
        y_train: (60000, ) array containing training image labels
        y_test: (10000, ) array containing testing image labels
    """
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    for x, y in list(itertools.product(["train", "test"], ["images", "labels"])):
        image_size = 28
        if x == "train":
            num_images = 60000
            images_path = "../data/train-images-idx3-ubyte.gz"
            labels_path = "../data/train-labels-idx1-ubyte.gz"
        else:
            num_images = 10000
            images_path = "../data/t10k-images-idx3-ubyte.gz"
            labels_path = "../data/t10k-labels-idx1-ubyte.gz"
        if y == "images":
            f = gzip.open(images_path, "r")
            f.read(16)
            buf = f.read(image_size * image_size * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(num_images, image_size * image_size)
            if x == "train":
                X_train = data
            else:
                X_test = data
        else:
            f = gzip.open(labels_path, "r")
            f.read(8)
            buf = f.read(num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            if x == "train":
                y_train = labels
            else:
                y_test = labels

    return X_train, X_test, y_train, y_test


def make_batches(
    data: np.ndarray, labels: np.ndarray, batch_size: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates batches from shuffled data.

    Args:
        data: (num_samples, num_features) array containing image data without labels
        labels: (num_samples, ) array containing image labels
        batch_size: number of samples in each batch

    Returns:
        list of tuples where each tuple contains a (batch_size, n_features) array
        containing image data and a (batch_size, ) array containing the corresponding labels
    """
    labels = labels[:, np.newaxis]
    concatenated = np.hstack((data, labels))
    np.random.shuffle(concatenated)
    new_data = concatenated[:, :-1]
    new_labels = concatenated[:, -1]
    breaks = [
        batch_size * i for i in range(1, (concatenated.shape[0] - 1) // batch_size + 1)
    ]
    split_data = np.array_split(new_data, breaks)
    split_labels = np.array_split(new_labels, breaks)
    batches = list(zip(split_data, split_labels))
    return batches


def display_mnist_image(image: np.ndarray):
    """
    Displays image of digit in the MNIST dataset

    Args:
        image: (784, ) array containing MNIST image
    """
    plt.imshow(image.reshape(28, 28))
    plt.show()
