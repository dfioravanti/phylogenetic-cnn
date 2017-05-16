#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import (
    Layer,
    Input,
    MaxPool2D,
    Reshape,
    Dense,
    Flatten,
    Dropout,
    Lambda
)
from keras.layers.merge import Concatenate
from keras.layers.convolutional import (
    Conv2D
)
from keras import backend as K
import tensorflow as tf


def _euclidean_distances(X):
    """  
    Considering the rows of X as vectors, compute the
    distance matrix between each pair of vectors. This is reimplementation 
    for Keras of sklearn.metrics.pairwise.euclidean_distances 
    
    :param coordinates: 
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


class Phcnn(Conv2D):

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
        super(Phcnn, self).__init__(
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


class Phngb(Layer):

    def __init__(self,
                 coordinates,
                 nb_neighbors,
                 nb_features,
                 **kwargs):

        super(Phngb, self).__init__(**kwargs)
        self.nb_neighbors = nb_neighbors
        self.nb_features = nb_features
        self.dist = _euclidean_distances(K.transpose(coordinates))

    def compute_output_shape(self, input_shape):
        return [(input_shape[0],
                 1,
                 input_shape[1] * self.nb_neighbors,
                 1)]

    def call(self, inputs):

        _, neighbor_indexes = tf.nn.top_k(-self.dist, k=self.nb_neighbors)

        output = K.expand_dims(inputs[:, 0], 1)
        for nth_neighbor in range(self.nb_neighbors):
            target_neighbor = neighbor_indexes[0, nth_neighbor]
            output = K.concatenate([output,
                                    K.expand_dims(inputs[:, target_neighbor], 1)
                                    ], axis=1)

        for feature in range(1, self.nb_features):
            for nth_neighbor in range(self.nb_neighbors):
                target_neighbor = neighbor_indexes[feature, nth_neighbor]
                output = K.concatenate([output,
                                        K.expand_dims(inputs[:, target_neighbor], 1)
                                        ], axis=1)

        output = K.expand_dims(output, 1)
        return K.expand_dims(output, 3)


def _slice_on_third(coord):

    def f(x):
        return x[:, :, coord]

    return Lambda(lambda x: f(x))


def _conv_block(conv, conv_crd, nb_neighbors, filters):

    xs = Reshape((conv.shape[2].value, conv.shape[3].value))(conv)
    crd = Reshape((conv_crd.shape[2].value, conv_crd.shape[3].value))(conv_crd)

    xs_sliced = _slice_on_third(0)(xs)
    crd_sliced = _slice_on_third(0)(crd)

    phngb = Phngb(coordinates=crd_sliced,
                  nb_neighbors=nb_neighbors,
                  nb_features=crd_sliced.shape[1])
    phcnn = Phcnn(nb_neighbors=nb_neighbors,
                  filters=filters)
    conv = phcnn(phngb(xs_sliced))
    conv_crd = phcnn(phngb(crd_sliced))

    for i in range(1, xs.shape[2].value):
        xs_sliced = _slice_on_third(i)(xs)
        crd_sliced = _slice_on_third(i)(crd)
        phngb = Phngb(coordinates=crd_sliced,
                      nb_neighbors=nb_neighbors,
                      nb_features=crd_sliced.shape[1])
        phcnn = Phcnn(nb_neighbors=nb_neighbors,
                      filters=filters)
        conv = Concatenate()([conv, phcnn(phngb(xs_sliced))])
        conv_crd = Concatenate()([conv_crd, phcnn(phngb(crd_sliced))])

    return conv, conv_crd


class PhcnnBuilder(object):

    @staticmethod
    def build(nb_coordinates, nb_features, nb_outputs, nb_neighbors=2):
        """Builds a custom ResNet like architecture.
        Args:
            nb_coordinates: The number of coordinates 
            nb_features: The number of features
            nb_outputs: The number of outputs at final softmax layer
            nb_neighbors: the number of phyloneighboors to be considered in the convolution
        Returns:
            The keras `Model`.
        """

        nb_filters = 1

        x = Input(shape=(nb_features,), name="xs_input", dtype='float64')
        coordinates = Input(shape=(nb_coordinates, nb_features), name="coordinates_input",  dtype='float64')
        coord = coordinates[0]

        phngb = Phngb(coordinates=coord,
                      nb_neighbors=nb_neighbors,
                      nb_features=coord.shape[1])
        phcnn = Phcnn(nb_neighbors=nb_neighbors,
                      filters=nb_filters)
        conv1 = phcnn(phngb(x))
        conv_crd1 = phcnn(phngb(coord))

        conv2, conv_crd2 = _conv_block(conv1, conv_crd1, nb_neighbors=nb_neighbors, filters=nb_filters)
        conv3, _ = _conv_block(conv2, conv_crd2, nb_neighbors=nb_neighbors, filters=nb_filters)

        max = MaxPool2D(pool_size=(1, 2), padding="valid")(conv3)
        flatt = Flatten()(max)
        drop = Dropout(0, 1)(Dense(units=64)(flatt))
        output = Dense(units=nb_outputs, kernel_initializer="he_normal",
                       activation="softmax", name='output')(drop)

        return Model(inputs=[x, coordinates], outputs=output)
