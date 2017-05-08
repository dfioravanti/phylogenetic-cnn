#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    ZeroPadding2D,
    Cropping2D
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
)
import numpy as np
from sklearn.metrics import euclidean_distances
import tensorflow as tf


def _phyloneighbors(xs, coordinates, k):
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
    dist = euclidean_distances(coordinates.eval().transpose())

    # neighbor_indexes[i,j] is index of the jth closest feature to the feature i
    neighbor_indexes = np.zeros((number_of_features, number_of_features), dtype='int')
    for i in range(number_of_features):
        neighbor_indexes[i] = np.argsort(dist[i])

    output = tf.zeros((number_of_samples, number_of_features * k))
    for feature in range(number_of_features):
        for nth_neighbor in range(k):
            target_feature = (k * feature) + nth_neighbor
            target_neighbor = neighbor_indexes[feature, nth_neighbor]

            output[:, target_feature] = xs[: target_neighbor]

    return tf.reshape(output, (number_of_samples, 1, 1, output.shape[1]))


def _phylo_convolution_relu(**conv_params):
    """
        Helper to build a phyloneighboor -> conv -> relu block
    """

    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    subsample = conv_params["subsample"]
    num_neighbors = conv_params["num_neighbors"]

    def f(inputs):

        xs = inputs[:, 0, :, 0]
        coordinates = inputs[:, 1, 0, :]

        neighbors = _phyloneighbors(xs, coordinates, num_neighbors)
        conv = Conv2D(input_shape=(1, 1, xs.shape[1] * num_neighbors),
                      filters=filters, kernel_size=kernel_size,
                      border_mode='valid', activation='relu', subsample=subsample)(neighbors)
        return conv

    return f


class PhcnnBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, num_neighbors=2):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_nb_features, nb_coordinates)
            num_outputs: The number of outputs at final softmax layer
            num_neighbors: the number of phyloneighboors to be considered in the convolution
        Returns:
            The keras `Model`.
        """

        inputs = Input(shape=input_shape)
        pyloconv1 = _phylo_convolution_relu(filters=4, kernel_size=(1, num_neighbors),
                                            subsample=(1, num_neighbors),
                                            num_neighboors=num_neighbors)(inputs)
        padd1 = ZeroPadding2D(padding=(0, 1))(pyloconv1)
        cropp1 = Cropping2D(cropping=((0, 0), (1, 0)))(padd1)
        max1 = MaxPooling2D(pool_size=(1, 2), border_mode="valid")(cropp1)
        flatt1 = Flatten()(max1)
        drop1 = Dense(activation="dropout")(flatt1)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(drop1)

        model = Model(inputs=inputs, outputs=dense)
        return model
