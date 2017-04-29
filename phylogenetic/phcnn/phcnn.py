#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .globalsettings import GlobalSettings
from .utils import load_datafile


def get_data(datafile, label_datafile, coordinates_datafile):

    """
    
    Load all the required data for phcnn from files. Three files are required,
        * datafile : the first row contains the names of the features,
                     the first column contains the names for the samples,
                     the remaining entries are the data
        * labels_datafile : Every entry i corresponds to the label associated with the ith sample in the datafile
        * coordinate_datafile: the first row contains the names of the features (we discard this since is redundant),
                               the first column contains the names for the coordinate (1,...,n for some n),
                               the remaining entries are the coordinates
       
    Return: A dictionary with all the data.
    
    """

    feature_names, sample_names, xs = load_datafile(datafile)
    ys = np.loadtxt(label_datafile, dtype=np.int)
    _, coordinate_names, coordinates = load_datafile(coordinates_datafile)

    return { 'feature_names' : feature_names,
             'sample_names' : sample_names,
             'xs' : np.copy(xs),
             'ys' : np.copy(ys),
             'coordinate_names' : coordinate_names,
             'coordinates' : coordinates}
