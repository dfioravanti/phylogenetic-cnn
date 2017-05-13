#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import (
    Layer,
    Input,
    MaxPool2D,
    Reshape,
    Permute,
    Activation,
    Dense,
    Flatten,
    Dropout,
)
from keras.layers.convolutional import (
    Conv2D
)
from keras.engine import InputSpec
from keras import backend as K
import tensorflow as tf

def _euclidean_distances(X):
    """  
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
    # -- ??
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
        #self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return [(input_shape[0],
                 1,
                 input_shape[1] * self.nb_neighbors,
                 1)]

    def call(self, inputs):

        _, neighbor_indexes = tf.nn.top_k(-self.dist, k=self.nb_neighbors)

        output = K.expand_dims(inputs[:, 0], 1)

        for feature in range(0, self.nb_features):
            for nth_neighbor in range(self.nb_neighbors):
                if not (feature == nth_neighbor == 0):
                    target_neighbor = neighbor_indexes[feature, nth_neighbor]
                    output = K.concatenate([output,
                                            K.expand_dims(inputs[:, target_neighbor], 1)
                                            ], axis=1)

        output = K.expand_dims(output, 1)
        return K.expand_dims(output, 3)


class PhcnnBuilder(object):

    @staticmethod
    def build(nb_coordinates, nb_features, nb_outputs, nb_neighbors=2):
        """Builds a custom ResNet like architecture.
        Args:
            xs_shape: The shape of the xs tensor in the (nb_samples, nb_features)
            coordinates_shape: The shape of the xs tensor in the (nb_samples, nb_coordinates)
            nb_outputs: The number of outputs at final softmax layer
            nb_neighbors: the number of phyloneighboors to be considered in the convolution
        Returns:
            The keras `Model`.
        """

        x = Input(shape=(nb_features,), name="xs_input", dtype='float64')
        coordinates = Input(shape=(nb_coordinates, nb_features), name="coordinates_input",  dtype='float64')
        coord = coordinates[0]

        phngb = Phngb(coordinates=coord,
                      nb_neighbors=8,
                      nb_features=coord.shape[1])
        phcnn = Phcnn(nb_neighbors=nb_neighbors,
                      filters=8)
        x1 = phcnn(phngb(x))
        #x1 = Reshape((temp.shape[2].value, temp.shape[3].value))(temp)
        #coord1 = phcnn(phngb(coord))
        max = MaxPool2D(pool_size=(1, 2), padding="valid")(x1)
        flatt = Flatten()(max)
        drop = Dropout(0, 1)(Dense(units=256)(flatt))
        output = Dense(units=nb_outputs, kernel_initializer="he_normal",
                       activation="softmax", name='output')(drop)

        return Model(inputs=[x, coordinates], outputs=output)
