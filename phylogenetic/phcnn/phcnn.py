#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import (
    Layer,
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
from keras.layers import (
    activations,
    initializers,
    regularizers,
    constraints
)
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.engine.topology import _to_list
from keras import backend as K
import numpy as np
from sklearn.metrics import euclidean_distances
import tensorflow as tf


class Phcnn(Layer):
    """
    """

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
        super(Phcnn, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (1, nb_neighbors)
        self.strides = (1, nb_neighbors)
        self.padding = padding
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        #self.input_spec = InputSpec(4)

    def build(self, input_shape):

        kernel_shape = self.kernel_size + (1, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        super(Phcnn, self).build(input_shape)

    def call(self, inputs):
        xs = inputs[0]
        coordinates = inputs[1]

        outputs = K.conv2d(
            xs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        new_coordinates = K.conv2d(
            coordinates,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        return [self.activation(outputs), new_coordinates]

    def compute_output_shape(self, input_shape):
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)


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
    d = K.maximum(d, K.constant(0))
    # -- ??
    return K.sqrt(d)


class Phngb(Layer):

    def __init__(self,
                 nb_neighbors,
                 **kwargs):

        super(Phngb, self).__init__(**kwargs)
        self.nb_neighbors = nb_neighbors
        #self.input_spec = InputSpec(ndim=3)

    def compute_mask(self, inputs, mask=None):
        super_mask = super(Phngb, self).compute_mask(inputs, mask)

        if super_mask is None and len(inputs) > 1:
            return [None for i in range(len(inputs))]

        return super_mask

    def compute_output_shape(self, input_shape):
        xs_shape = input_shape[0]
        coordinates_shape = input_shape[1]
        return [(xs_shape[0],
                 1,
                 xs_shape[1] * self.nb_neighbors,
                 1),
                (coordinates_shape[0],
                 1,
                 coordinates_shape[1] * self.nb_neighbors,
                 1)]

    def call(self, inputs):
        xs = inputs[0]
        coordinates = inputs[1]

        nb_features = coordinates.shape[1]
        # dist[i,j] is the distance between the ith feature and the jth feature
        dist = _euclidean_distances(K.transpose(coordinates))
        # neighbor_indexes[i,j] is index of the jth closest feature to the feature i
        _, neighbor_indexes = tf.nn.top_k(-dist, k=self.nb_neighbors)

        expanded_xs = K.expand_dims(xs[:, 0], 1)
        expanded_coordinates = K.expand_dims(coordinates[:, 0], 1)

        for feature in range(0, nb_features):
            for nth_neighbor in range(self.nb_neighbors):
                if not (feature == nth_neighbor == 0):
                    target_neighbor = neighbor_indexes[feature, nth_neighbor]
                    expanded_xs = K.concatenate([expanded_xs,
                                                 K.expand_dims(xs[:, target_neighbor], 1)
                                                 ], axis=1)
                    expanded_coordinates = K.concatenate([expanded_coordinates,
                                                          K.expand_dims(coordinates[:, target_neighbor], 1)
                                                          ], axis=1)

        expanded_xs = K.expand_dims(expanded_xs, 1)
        expanded_xs = K.expand_dims(expanded_xs, 3)
        expanded_coordinates = K.expand_dims(expanded_coordinates, 1)
        expanded_coordinates = K.expand_dims(expanded_coordinates, 3)

        return [expanded_xs, expanded_coordinates]


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

        x = Input(shape=(nb_features,), name="xs_input")
        coordinates = Input(shape=(nb_coordinates, nb_features), name="coordinates_input")
        coord = coordinates[0]

        phngb = Phngb(nb_neighbors=nb_neighbors)([x, coord])
        #phyloconv1 = Phcnn(filters=4, nb_neighbors=nb_neighbors)(phngb)
        #get_conv_weigth = K.function([], [phyloconv1.get_weights()])
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

        return Model(inputs=[x, coordinates], outputs=phngb)
