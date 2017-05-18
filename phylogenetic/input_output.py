import numpy as np
from phcnn.utils import load_datafile
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
    :return: A dictionary with all the data required by phcnn
    
    """

    feature_names, sample_names, Xs = load_datafile(datafile)
    ys = np.loadtxt(labels_datafile, dtype=np.int)

    _, _, Xs_test = load_datafile(test_datafile)
    ys_test = np.loadtxt(test_label_datafile, dtype=np.int)
    _, coordinate_names, coordinates = load_datafile(coordinates_datafile)

    all_coordinates = np.empty((Xs.shape[0],) + coordinates.shape, dtype=np.float64)
    all_coordinates[0] = coordinates

    inputs = Bunch()
    inputs.feature_names = feature_names
    inputs.sample_names = sample_names
    inputs.xs = Xs
    inputs.ys = ys
    inputs.Xs_test = Xs_test
    inputs.ys_test = ys_test
    inputs.coordinate_names = coordinate_names
    inputs.coordinates = all_coordinates
    inputs.nb_samples = Xs.shape[0]
    inputs.nb_features = Xs.shape[1]
    inputs.nb_coordinates = coordinates.shape[0]
    inputs.nb_classes = len(np.unique(ys))
    return inputs

