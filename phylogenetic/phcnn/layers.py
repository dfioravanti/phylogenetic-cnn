#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Layer, Lambda
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D

from keras import backend as K
import tensorflow as tf


def _euclidean_distances(X):
    """  
    Considering the rows of X as vectors, compute the
    distance matrix between each pair of vectors. This is reimplementation 
    for Keras of sklearn.metrics.pairwise.euclidean_distances 
    
    :param X: 
    :return: 
    """

    Y = K.transpose(X)
    XX = K.expand_dims(K.sum(K.square(X), axis=1), 0)
    YY = K.transpose(XX)

    d = K.dot(X, Y)
    d *= -2
    d += XX
    d += YY
    d = K.maximum(d, K.constant(0, dtype=d.dtype))

    return K.sqrt(d)


class PhyloConv2D(Conv2D):

    def __init__(self,
                 nb_neighbors,
                 filters,
                 activation='relu',
                 padding='valid',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(PhyloConv2D, self).__init__(
            filters=filters,
            kernel_size=(1, nb_neighbors),
            strides=(1, nb_neighbors),
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
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
        crd = K.reshape(coordinates, (coordinates.shape[0].value, coordinates.shape[2].value))
        self.dist = _euclidean_distances(K.transpose(crd))

    def compute_output_shape(self, input_shape):
        return [(input_shape[0],
                 input_shape[1],
                 input_shape[2] * self.nb_neighbors,
                 input_shape[3])]

    def call(self, inputs, **kwargs):

        # TODO: Explain Trick!
        _, neighbor_indexes = tf.nn.top_k(-self.dist, k=self.nb_neighbors)

        def _gather_along_axis(data, indices, axis=0):
            """
            Adapted from this function 
            on Github: github.com/tensorflow/tensorflow/issues/206
            
            :param self: 
            :param data: 
            :param indices: 
            :param axis: 
            :return: 
            """
            if axis == 0:
                return K.gather(data, indices)
            rank = data.shape.ndims
            perm = [axis] + list(range(1, axis)) + [0] + list(range(axis + 1, rank))
            gather = K.gather(K.permute_dimensions(data, perm), indices)
            shapes = [s.value if s.value else -1 for s in gather.shape]
            shapes = tuple([shapes[0] * shapes[1]] + shapes[2:])
            gather = K.reshape(gather, shapes)
            out = K.permute_dimensions(gather, perm)
            return out

        # Add all the other features and their neighbors
        target_neighbors = neighbor_indexes[0: self.nb_features, 0:self.nb_neighbors]
        output = _gather_along_axis(inputs, target_neighbors, axis=2)

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

    def ConvFilterLayer(coord):
        def get_conv_filter(x):
            return K.expand_dims(x[:, :, :, coord], 3)
        return Lambda(lambda x: get_conv_filter(x))

    Xs_sliced = ConvFilterLayer(0)(Xs)
    Crd_sliced = ConvFilterLayer(0)(Crd)

    phngb = PhyloNeighbours(coordinates=Crd_sliced,
                            nb_neighbors=nb_neighbors,
                            nb_features=nb_features)
    phcnn = PhyloConv2D(nb_neighbors=nb_neighbors,
                        filters=filters)
    Xs_new = phcnn(phngb(Xs_sliced))
    Crd_new = phcnn(phngb(Crd_sliced))

    for i in range(1, Xs.shape[3].value):
        Xs_sliced = ConvFilterLayer(i)(Xs)
        Crd_sliced = ConvFilterLayer(i)(Crd)
        phngb = PhyloNeighbours(coordinates=Crd_sliced,
                                nb_neighbors=nb_neighbors,
                                nb_features=nb_features)
        phcnn = PhyloConv2D(nb_neighbors=nb_neighbors,
                            filters=filters)
        Xs_new = Concatenate()([Xs_new, phcnn(phngb(Xs_sliced))])
        Crd_new = Concatenate()([Crd_new, phcnn(phngb(Crd_sliced))])

    return Xs_new, Crd_new
