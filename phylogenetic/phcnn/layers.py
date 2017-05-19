#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Layer
from keras.layers.convolutional import Conv1D

from keras import backend as K
import tensorflow as tf


def _transpose_on_first_two_axes(X):
    perm = [0, 2, 1]
    return K.permute_dimensions(K.transpose(K.permute_dimensions(X, perm)), perm)


def _dot(X, Y):

    perm = [2, 0, 1]
    inv_perm = [1, 2, 0]

    X_perm = K.permute_dimensions(X, perm)
    Y_perm = K.permute_dimensions(Y, perm)

    dot = K.batch_dot(X_perm, Y_perm)

    return K.permute_dimensions(dot, inv_perm)


def _euclidean_distances(X):
    """  
    Considering the rows of X as vectors, compute the
    distance matrix between each pair of vectors. This is reimplementation 
    for Keras of sklearn.metrics.pairwise.euclidean_distances 
    
    :param X: 
    :return: 
    """

    Y = X
    X = _transpose_on_first_two_axes(X)

    XX = K.expand_dims(K.sum(K.square(X), axis=1), 0)
    YY = _transpose_on_first_two_axes(XX)

    d = _dot(X, Y)
    d *= -2
    d += XX
    d += YY
    d = K.maximum(d, K.constant(0, dtype=d.dtype))

    # we do not need the square root for the ranking so we return just d
    return d


class PhyloConv(Conv1D):

    def __init__(self,
                 nb_neighbors,
                 filters,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None):
        super(PhyloConv, self).__init__(
            filters=filters,
            kernel_size=nb_neighbors,
            strides=nb_neighbors,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )


class PhyloNeighbours(Layer):

    def __init__(self,
                 coordinates,
                 nb_neighbors,
                 nb_features,
                 **kwargs):

        super(PhyloNeighbours, self).__init__(**kwargs)
        self.nb_neighbors = nb_neighbors
        self.nb_features = nb_features
        self.dist = _euclidean_distances(coordinates)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0],
                 input_shape[1] * self.nb_neighbors,
                 input_shape[2])]

    def _top_k(self, dist, k):
        # TODO: Explain Trick!

        perm = [0, 2, 1]

        _, index = tf.nn.top_k(K.permute_dimensions(dist, perm), k=k)
        return K.permute_dimensions(index, perm)


    def _gather_along_axis(self, data, indices, axis=0):
        """
        Adapted from this function 
        on Github: github.com/tensorflow/tensorflow/issues/206

        :param self: 
        :param data: 
        :param indices: 
        :param axis: 
        :return: 
        """

        # features = indices.shape[0].value
        # nb = indices.shape[1].value
        #
        # permuted = K.permute_dimensions(data, [1, 2, 0])
        #
        # tiled = K.tile(permuted, [nb, 1, 1])
        # shapes = (features, nb, tiled.shape[1].value, tiled.shape[2].value)
        # shapes = [s if s else -1 for s in shapes]
        # resh = K.reshape(tiled, shapes)
        #
        # if K.backend() == 'theano':
        #     g = K.gather(resh, indices)
        # else:
        #     g = tf.gather_nd(resh, indices)
        #
        # return g





        # if not axis:
        #     return tf.gather(data, indices)
        # rank = data.shape.ndims
        # perm = [0, 1, 2]
        # temp = tf.gather_nd(tf.transpose(data, perm), indices)
        # output = tf.transpose(temp, perm)
        # return output

        perm = [1, 0]

        output = K.gather(K.permute_dimensions(data[:, :, 0], perm), indices[:, :, 0])
        shapes = [s.value if s.value else -1 for s in output.shape]
        shapes = tuple([shapes[0] * shapes[1]] + [shapes[2], 1])
        output = K.reshape(output, shapes)
        for i in range(1, data.shape[2].value):
            gather = K.gather(K.permute_dimensions(data[:, :, i], perm), indices[:, :, i])
            shapes = [s.value if s.value else -1 for s in gather.shape]
            shapes = tuple([shapes[0] * shapes[1]] + [shapes[2], 1])
            gather = K.reshape(gather, shapes)
            output = K.concatenate([output, gather])

        out = K.permute_dimensions(output, [1, 0, 2])
        return out

    def call(self, inputs, **kwargs):

        neighbor_indexes = self._top_k(-self.dist, k=self.nb_neighbors)

        # Add all the other features and their neighbors
        target_neighbors = neighbor_indexes[0: self.nb_features, 0:self.nb_neighbors, :]
        output = self._gather_along_axis(inputs, target_neighbors, axis=1)

        return output


def _conv_block(Xs, Crd, nb_neighbors, nb_features, filters):

    """
    
    :param Xs: 
    :param Crd: 
    :param nb_neighbors: 
    :param nb_features: 
    :param filters: 
    :return: 
    """

    phngb = PhyloNeighbours(coordinates=Crd,
                            nb_neighbors=nb_neighbors,
                            nb_features=nb_features)
    phcnn = PhyloConv(nb_neighbors=nb_neighbors,
                      filters=filters)
    Xs_new = phcnn(phngb(Xs))
    Crd_new = phcnn(phngb(Crd))

    return Xs_new, Crd_new
