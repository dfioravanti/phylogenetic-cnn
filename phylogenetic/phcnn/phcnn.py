#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras.optimizers import SGD, RMSprop, Adam

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing


from .globalsettings import GlobalSettings
from .utils import load_datafile
from .scaling import norm_l2


class OptimizerNotFound(Exception):
    pass


class ScalingNotFound(Exception):
    pass


def __configure_optimizer(selected_optimizer):
    """
    Configure a optimizer. It raise an exception if a wrong optimizer name is passed.

    :param selected_optimizer: string containing the name of the optimizer that we want to configure 
    :return: The optimizer and a dictionary containing the name and the parameter of the optimizer 
   """

    if selected_optimizer == 'adam':
        epsilon = 1e-08
        lr = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        optimizer_configuration = {'name': 'adam',
                                   'lr': lr,
                                   'beta_1': beta_1,
                                   'beta_2': beta_2,
                                   'epsilon': epsilon
                                   }
    elif selected_optimizer == 'rmsprop':
        epsilon = 1e-08
        lr = 0.001
        rho = 0.9
        optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon)
        optimizer_configuration = {'name': rmsprop,
                                   'lr': lr,
                                   'rho': rho,
                                   'epsilon': epsilon}
    elif selected_optimizer == 'sgd':
        nesterov = True
        lr = 0.001
        momentum = 0.9
        decay = 1e-06
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        optimizer_configuration = {'name': 'sgd',
                                   'lr': lr,
                                   'momentum': momentum,
                                   'decay': decay,
                                   'nesterov': nesterov}
    else:
        raise OptimizerNotFound("The only supported optimizer are adam, rmsprop, sgd")

    return optimizer, optimizer_configuration


def __apply_scaling(xs_train, xs_test, selected_scaling):
    """
    Apply the scaling to the input data AFTER they have been divided in 
    (training, testing) during a cross validation. It raise an exception 
    if a wrong scaler name is passed.
    :param xs_train: Training data
    :param xs_test: Test data
    :param selected_scaling: a string representing the select scaling
    :return: (xs_training, xs_test) scaled according to the scaler. 
    """
    if selected_scaling == 'norm_l2':
        xs_train_scaled, m_train, r_train = norm_l2(xs_train)
        xs_test_scaled, _, _ = norm_l2(xs_test, m_train, r_train)
    elif selected_scaling == 'std':
        scaler = preprocessing.StandardScaler(copy=False)
        xs_train_scaled = scaler.fit_transform(xs_train)
        xs_test_scaled = scaler.transform(xs_test)
    elif selected_scaling == 'minmax':
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        xs_train_scaled = scaler.fit_transform(xs_train)
        xs_test_scaled = scaler.transform(xs_test)
    elif selected_scaling == 'minmax0':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
        xs_train_scaled = scaler.fit_transform(xs_train)
        xs_test_scaled = scaler.transform(xs_test)
    else:
        raise ScalingNotFound("The only supported scaling are norm_l2, std, minmax, minmax0")

    return xs_train_scaled, xs_test_scaled


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


def phyloneighbors(xs, coordinates, k):
    """   
    :param xs: numpy array with shape (number of samples, number of features) where xs[i,j] is the jth feature of the
               ith sample.
               ith row = ith sample and jth column = jth feature.
    :param coordinates: numpy array with shape (number of coordinates, number of features) where coordinate[i,j] is the
           ith coordinate of the jth feature obtained with a MDS procedure. 
           ith row = ith coordinate of the features and jth column = jth feature.
    :param k: Number of desired neighbors. 
    :return: numpy array where every feature j in xs we add (k-1) columns denoting the k closest neighbors.
             Where closest is computed respect to the coordinates using the the euclidean distance.    
    """

    number_of_samples = xs.shape[0]
    number_of_features = xs.shape[1]

    # dist[i,j] is the distance between the ith feature and the jth feature
    dist = euclidean_distances(coordinates.transpose())

    # neighbor_indexes[i,j] is index of the jth closest feature to the feature i
    neighbor_indexes = np.zeros((number_of_features, number_of_features), dtype='int')
    for i in range(number_of_features):
        neighbor_indexes[i] = np.argsort(dist[i])

    output = np.zeros((number_of_samples, number_of_features * k))
    for feature in range(number_of_features):
        for nth_neighbor in range(k):
            target_feature = (k * feature) + nth_neighbor
            target_neighbor = neighbor_indexes[feature, nth_neighbor]

            output[:, target_feature] = xs[: target_neighbor]

    return np.reshape(output, (number_of_samples, 1, 1, output.shape[1]))


def data():
    """
    Data providing function required by hyperas:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    return get_data(GlobalSettings.datafile,
                    GlobalSettings.labels_datafile,
                    GlobalSettings.coordinates_datafile)
