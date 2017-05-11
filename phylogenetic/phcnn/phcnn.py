#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    ZeroPadding2D,
    Cropping2D,
    Lambda
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
)
from keras.layers.merge import add
from keras import backend as K
import numpy as np
from sklearn.metrics import euclidean_distances
import tensorflow as tf


# TODO: trash it
def gather_along_axis(data, indices, axis=0):
  if not axis:
    return K.gather(data, indices)
  rank = data.shape.ndims
  perm = [axis] + list(range(1, axis)) + [0] + list(range(axis + 1, rank))
  return tf.transpose(K.gather(tf.transpose(data, perm=perm), indices), perm=perm)





def _euclidean_distances(X):
    """
    
    :param coordinates: 
    :return: 
    """

    Y = K.transpose(X)
    XX = K.sqrt(K.sum(K.square(X), axis=1))
    YY = K.transpose(XX)
    d = K.dot(X, Y)
    d *= -2
    d += XX
    d += YY
    d = K.maximum(d, K.constant(0))
    # -- ??
    return K.sqrt(d)


def _phngb(coordinates, nb_neighbors):
    """
    Helper to build a phylo_neighbors layer
    :param coordinates: a tensor with shape (number of coordinates, number of features)
                        where coordinate[i,j] is the ith coordinate of the jth feature obtained with a MDS procedure.
                        ith row = ith coordinate of the features and jth column = jth feature.
    :param nb_neighbors: number of neighbor that will computed    
    """

    def f(xs):
        """
        :param xs: a tensor with shape (batch_size, nb_features) containing the current features  
        :return: a keras Lambda layer that applies returns a tensor where every entry of x is followed by its 
                 nb_neighbors closes neighbors. 
        """

        nb_features = coordinates.shape[1]
        # dist[i,j] is the distance between the ith feature and the jth feature
        dist = _euclidean_distances(coordinates.transpose())
        # neighbor_indexes[i,j] is index of the jth closest feature to the feature i
        neighbor_indexes = np.zeros((nb_features, nb_features), dtype='int')
        for i in range(nb_features):
            neighbor_indexes[i] = np.argsort(dist[i])

        output = (tf.slice(xs, [0, 0], [-1, 1]))

        for feature in range(1, nb_features):
            for nth_neighbor in range(nb_neighbors):
                target_neighbor = neighbor_indexes[feature, nth_neighbor]
                output = K.concatenate([output,
                                        tf.slice(xs, [0, target_neighbor], [-1, 1])
                                        ], axis=1)

        return tf.reshape(output, (1, nb_features * nb_neighbors, 1))

    return Lambda(lambda x: f(x))


def _phylo_convolution(**conv_params):
    """
        Helper to build a phyloneighboor -> conv -> relu block
    """

    filters = conv_params["filters"]
    nb_neighbors = conv_params["nb_neighbors"]

    def f(xs):

        return Conv2D(input_shape=(1, xs.shape[1], 1),
                      filters=filters, kernel_size=(1, nb_neighbors),
                      padding='valid', activation='relu', strides=(1, nb_neighbors))(xs)

    return f


class PhcnnBuilder(object):

    @staticmethod
    def build(nb_features, coordinates, nb_outputs, nb_neighbors=2):
        """Builds a custom ResNet like architecture.
        Args:
            xs_shape: The shape of the xs tensor in the (nb_samples, nb_features)
            coordinates_shape: The shape of the xs tensor in the (nb_samples, nb_coordinates)
            nb_outputs: The number of outputs at final softmax layer
            nb_neighbors: the number of phyloneighboors to be considered in the convolution
        Returns:
            The keras `Model`.
        """

        x = Input(shape=(nb_features,), name="xs_input")
        phngb = _phngb(coordinates, nb_neighbors)(x)
        pyloconv1 = _phylo_convolution(filters=4, nb_neighbors=nb_neighbors)(phngb)
        # padd1 = ZeroPadding2D(padding=(0, 1))(pyloconv1)
        # cropp1 = Cropping2D(cropping=((0, 0), (1, 0)))(padd1)
        # max1 = MaxPooling2D(pool_size=(1, 2), border_mode="valid")(cropp1)
        # flatt1 = Flatten()(max1)
        # drop1 = Dense(activation="dropout")(flatt1)
        # dense = Dense(units=nb_outputs, kernel_initializer="he_normal",
        #               activation="softmax", name='output')(drop1)
        #
        # model = Model(inputs=['xs_input', 'coordinates_input'], outputs=dense)

        #flatt = Flatten()(phngb)
        #dense = Dense(units=2, kernel_initializer="he_normal",
        #             activation="relu", name='output')(phngb)

        #y = Lambda(lambda t: tf.slice(t, [0, 0], [-1, 32]))(phngb)

        return Model(inputs=x, outputs=pyloconv1)
