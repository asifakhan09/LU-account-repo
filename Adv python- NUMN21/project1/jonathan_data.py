import gzip
import pickle
import numpy as np

def load_mnist(path = "mnist.pkl.gz"):
    """
    Loads the MNIST dataset, from gzipped pickle file.
    Returns: (train_set, valid_set, test_set)
            Each set is a tuple (x, y) where x is of shape (n_samples, 784) 
            and y is an array of itegers (digit labels 0-9)
    """

    with gzip.open(path) as f: 
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    return train_set, valid_set, test_set

def convert_label(labels, num_classes=10):
    """
    Converts digit labels (0-9) into one-hot vectors for classification,
    example: 3 -> [0,0,0,1,0,0,0,0,0,0].
    """
    y = np.zeros((labels.size, num_classes))
    y[np.arange(labels.size), labels] = 1.0
    return y