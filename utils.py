import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch


def get_data(datafile, labels_datafile, coordinates_datafile,
             test_datafile, test_label_datafile):
    """   
    :param datafile: the first row contains the names of the features,
                     the first column contains the names for the samples,
                     the remaining entries are the data
    :param labels_datafile: Every entry i corresponds to the label associated with the ith sample in the datafile
    :param coordinates_datafile: the first row contains the names of the features (we discard this since is redundant),
                               the first column contains the names for the coordinate (1,...,n for some n),
                               the remaining entries are the coordinates
    :param test_datafile: same structure as datafile, but it contains the test data
    :param test_label_datafile: same structure as labels_test_datafile, but it contains the test labels
    :return: A Bunch with all the data required by phcnn
    
    """

    feature_names, sample_names, Xs = load_datafile(datafile)
    ys = np.loadtxt(labels_datafile, dtype=np.int)

    _, _, Xs_test = load_datafile(test_datafile)
    ys_test = np.loadtxt(test_label_datafile, dtype=np.int)
    _, coordinate_names, coordinates = load_datafile(coordinates_datafile)

    # Keras needs that all the batch sizes are the same. And it is impossible at the
    # moment to pass a variable tot the model. Since coordinates is batch independent
    # we need to expand it to match the shape of Xs but only the first "face" of coordinates
    # is actually used all the rest is basically just pudding. This is a huge waste of memory but is
    # the least worse solution we found.
    all_coordinates = np.empty((Xs.shape[0],) + coordinates.shape, dtype=np.float64)
    for i in range(Xs.shape[0]):
        all_coordinates[i] = coordinates

    inputs = Bunch()
    inputs.feature_names = feature_names
    inputs.sample_names = sample_names
    inputs.training_data = Xs
    inputs.targets = ys
    inputs.test_data = Xs_test
    inputs.test_targets = ys_test
    inputs.coordinate_names = coordinate_names
    inputs.coordinates = all_coordinates
    inputs.nb_samples = Xs.shape[0]
    inputs.nb_features = Xs.shape[1]
    inputs.nb_coordinates = coordinates.shape[0]
    inputs.nb_classes = len(np.unique(ys))
    return inputs


def load_datafile(filepath, sep='\t', dtype=np.float):

    """
    Load a specific file which we assume have the following structure:
        the first row contains the names of the features,
        the first column contains the names for the samples,
        the remaining entries are the data with type dtype.
    """
    df = pd.read_csv(filepath, sep=sep, header=0, index_col=0)
    sample_names = df.index.tolist()
    feature_names = df.columns.tolist()
    data = df.as_matrix().astype(dtype=dtype)
    return feature_names, sample_names, data


def to_list(e):
    """Utility function to convert to 
    a Python list any input expected to be so."""
    if isinstance(e, list):
        return e
    return [e]
