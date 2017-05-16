from phcnn import settings
import numpy as np
from phcnn.utils import load_datafile


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
       
    :return: A dictionary with all the data required by phcnn.
    
    """

    # FIXME: Add missing parameters in the Docstring!!

    feature_names, sample_names, Xs = load_datafile(datafile)
    ys = np.loadtxt(labels_datafile, dtype=np.int)

    _, _, Xs_test = load_datafile(test_datafile)
    ys_test = np.loadtxt(test_label_datafile, dtype=np.int)
    _, coordinate_names, coordinates = load_datafile(coordinates_datafile)

    all_coordinates = np.zeros((Xs.shape[0],) + coordinates.shape)
    for i in range(Xs.shape[0]):
        all_coordinates[i] = coordinates

    return {'feature_names': feature_names,
            'sample_names': sample_names,
            'xs': Xs,
            'ys': ys,
            'Xs_test': Xs_test,
            'ys_test': ys_test,
            'coordinate_names': coordinate_names,
            'coordinates': all_coordinates,
            'nb_samples': Xs.shape[0],
            'nb_features': Xs.shape[1],
            'nb_coordinates': coordinates.shape[0],
            'nb_classes': len(np.unique(ys))
            }


