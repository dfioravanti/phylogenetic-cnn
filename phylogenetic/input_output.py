import argparse
import sys

import numpy as np

from phylogenetic.phcnn.utils import load_datafile


def create_parser():

    parser = argparse.ArgumentParser(
        description='Run a training experiment (10x5-CV fold) using Rectified Factor Networks,\
                    Support Vector Machines, Random Forests and/or Multilayer Perceptron.',
        fromfile_prefix_chars='@')

    parser.add_argument('DATAFILE', type=str, help='Training datafile')
    parser.add_argument('COORDINATES', type=str, help='Coordinates datafile')
    parser.add_argument('LABELSFILE', type=str, help='Sample labels')
    parser.add_argument('OUTDIR', type=str, help='Output directory')

    parser.add_argument('--scaling', dest='SCALING', type=str, choices=['norm_l2', 'std', 'minmax', 'minmax0'],
                        default='std', help='Scaling method (default: %(default)s)')
    parser.add_argument('--ranking', dest='RANK_METHOD', type=str, choices=['ReliefF', 'tree', 'KBest', 'random'],
                        default='ReliefF',
                        help='Feature ranking method: ReliefF, extraTrees, Anova F-score,\
                              random ranking (default: %(default)s)')
    parser.add_argument('--cv_k', type=np.int, default=5, help='Number of CV folds (default: %(default)s)')
    parser.add_argument('--cv_n', type=np.int, default=10, help='Number of CV cycles (default: %(default)s)')
    parser.add_argument('--reliefk', type=np.int, default=3,
                        help='Number of nearest neighbors for ReliefF (default: %(default)s)')
    parser.add_argument('--rfep', type=np.float, default=0.2,
                        help='Fraction of features to remove at each iteration in RFE\
                            (p=0 one variable at each step, p=1 naive ranking) (default: %(default)s)')
    parser.add_argument('--plot', action='store_true', help='Plot metric values over all training cycles')
    parser.add_argument('--tsfile', type=str, default=None, help='Validation datafile')
    parser.add_argument('--tslab', type=str, default=None, help='Validation labels, if available')
    parser.add_argument('--trials', type=int, default=None, help='Number of hypersearch trials.')
    parser.add_argument('--quiet', action='store_true', help='Run quietly (no progress info)')
    parser.add_argument('--allfeatures', action='store_true', help='Do not perform features step')

    return parser

def get_data(datafile,
             labels_datafile,
             coordinates_datafile,
             validation_datafile=None,
             validation_label_datafile=None):
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

    feature_names, sample_names, xs = load_datafile(datafile)
    ys = np.loadtxt(labels_datafile, dtype=np.int)

    if validation_datafile is not None and validation_label_datafile is not None:
        _, _, validation_xs = load_datafile(validation_datafile)
        validation_ys = np.loadtxt(validation_label_datafile, dtype=np.int)
    else:
        validation_xs = validation_ys = None

    _, coordinate_names, coordinates = load_datafile(coordinates_datafile)

    all_coordinates = np.zeros((xs.shape[0],) + coordinates.shape)
    for i in range(xs.shape[0]):
        all_coordinates[i] = np.copy(coordinates)

    return {'feature_names': feature_names,
            'sample_names': sample_names,
            'xs': np.copy(xs),
            'ys': np.copy(ys),
            'validation_xs': np.copy(validation_xs),
            'validation_ys': np.copy(validation_ys),
            'coordinate_names': coordinate_names,
            'coordinates': all_coordinates,
            'nb_samples': xs.shape[0],
            'nb_features': xs.shape[1],
            'nb_coordinates': coordinates.shape[0],
            'nb_classes': len(set(ys))
            }


if __name__ == '__main__':
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    print(args)