import argparse
import configparser
import mlpy
import numpy as np
import os
import sys

from phylogenetic.phcnn.utils import load_datafile
from phylogenetic.phcnn.globalsettings import GlobalSettings


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


def get_data(datafile, labels_datafile, coordinates_datafile):
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
    _, coordinate_names, coordinates = load_datafile(coordinates_datafile)

    return {'feature_names': feature_names,
            'sample_names': sample_names,
            'xs': np.copy(xs),
            'ys': np.copy(ys),
            'coordinate_names': coordinate_names,
            'coordinates': coordinates,
            'number_of_samples': xs.shape[0],
            'number_of_features': xs.shape[1]
            }


if __name__ == '__main__':
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    print(args)