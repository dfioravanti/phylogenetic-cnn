#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import InputSpec
from keras.layers.convolutional import Conv1D

from keras import backend as K
import tensorflow as tf
from theano import tensor as T


def _transpose_on_first_two_axes(X):
    """
    This function will permute each face of a 3D tensor separately.
    Suppose X = [A, B] where 
    A = [[1, 2], and B = [[5, 6],
         [3, 4]]          [7, 8]]
    Then the output will be [K.transpose(A), K.transpose(B)] where
    K.transpose(A) = [[1, 3], and  K.transpose(B) = [[5, 7],
                      [2, 4]]                        [6, 8]]
    Parameters
    ----------
    X: K.Tensor
        A 3D tensor

    Returns
    -------
    out: K.Tensor
        The 3D tensor with the permuted faces 
    
    
    """
    perm = [0, 2, 1]
    out = K.permute_dimensions(K.transpose(K.permute_dimensions(X, perm)), perm)
    return out


def _dot(X, Y):
    """
    Compute the matrix usual matrix multiplication between the faces of two tensors
    among the last axes. We assume shape X.shape = (a, b, k), Y.shape = (b, c, k).
    
    To do so we use K.batch_dot. We need to move the last axis in first position since
    K.batch_dot assumes that the shape is (batch_size, other_sizes:)
     
    Example:
        If X = [A, B] and Y = [C, D] with shape(A) = shape(B) = (a, b) and
        shape(C) = shape(D) = (b, c) the _dot(X, Y) = [(AC), (BD)] 
    
    Parameters
    ----------
    X: K.Tensor
    Y: K.Tensor

    Returns
    -------
    dot: K.Tensor
    """

    last_axis_first = [2, 0, 1]
    first_axis_last = [1, 2, 0]

    X_permuted = K.permute_dimensions(X, last_axis_first)
    Y_permuted = K.permute_dimensions(Y, last_axis_first)

    dot = K.permute_dimensions(K.batch_dot(X_permuted, Y_permuted),
                               first_axis_last)

    return dot


def euclidean_distances(X):
    """
    Considering the rows of every face of X as vectors, compute the
    distance matrix between each pair of vectors. This is reimplementation 
    for Keras of sklearn.metrics.pairwise.euclidean_distances. Compared to
    the sklearn implementation there are two important differences. First we 
    do not impose that distance[i,i] = 0 for all i, since the round error are
    irrelevant for us and we return the square of the distance as we need 
    the distances in order to create a ranking and 0 > a > b <=> a^2 > b^2. 
     
    Example:
        If X = [A, B] what this algorithm does is to return a tensor 
        T = [euclidean_distances(A)^2, euclidean_distances(B)^2]
        
    Parameters
    ----------
    X: K.Tensor

    Returns
    -------
    distance: K.Tensor

    """

    Y = X
    X = _transpose_on_first_two_axes(X)

    XX = K.expand_dims(K.sum(K.square(X), axis=1), 0)
    YY = _transpose_on_first_two_axes(XX)

    distance = _dot(X, Y)
    distance_shapes = K.int_shape(distance)
    distance *= -2
    distance += XX
    distance += YY
    distance = K.maximum(distance,
                         K.constant(0, dtype=distance.dtype))
    distance._keras_shape = distance_shapes
    return distance


def _gather_target_neighbors(data, indices):

    """
    This function extracts the correct indexes from data. We assume that data as shape
    (batch_size, nb_features, filters) and indices as shape (nb_feature, nb_neighbors, filters). 
    For a fixed filter k indices[i, j, k] = l is the index of the jth closes feature to the feature i. 
    So we need to retrieve data[:, l: k]. 
    
    To do so we need to use K.gather, since K.gather only works for the first axis we need to swap axis 0 and 1.
    After K.gather is applied we need to apply a reshape in order to go back to the original shape. Once all the
    data are retrieved we swap back axis 0 and 1. 
    
    Yes, we know is a mess but K.gather is the only API available at the moment.
    
    Parameters
    ----------
    data: K.Tensor
        Tensor of shape (batch_size, nb_features, filters)
    indices: K.Tensor
        Tensor of shape (nb_feature, nb_neighbors, filters)
    
    Returns
    -------
    out: K.Tensor
        Tensor of shape (batch_size, nb_features * nb_neighbors, filters)
    
    """
    indices_shape = K.int_shape(indices)
    data_shape = K.int_shape(data)
    nb_samples = int(data_shape[0]) if data_shape[0] else -1
    nb_features = int(indices_shape[0])
    nb_neighbours = int(indices_shape[1])
    nb_channels = int(data_shape[2])

    if K.backend() == 'tensorflow':
        perm = [1, 2, 0]
        data_nd = K.permute_dimensions(data, perm)
        indices = K.expand_dims(indices, 3)
        gather_neighbours = tf.gather_nd(data_nd, indices)
        gather_neighbours = gather_neighbours[:, :, :, 0]
        target_neighbours = K.reshape(gather_neighbours, shape=(nb_features*nb_neighbours,
                                                                nb_channels, nb_samples))
        target_neighbours = K.permute_dimensions(target_neighbours, [2, 0, 1])
    else:
        target_neighbours = data[:, indices]
        target_neighbours = target_neighbours[:, :, :, 0]
        target_neighbours = K.reshape(target_neighbours, shape=(nb_samples,
                                                                nb_features*nb_neighbours, nb_channels))
    return target_neighbours


def _top_k(dist, k):
    """
    This function is just a wrapper for top_k of TensorFlow. top_k assumes that the axis to be
    ordered is the last one and we need to order according to the second one. So we apply a 
    permute before and after.
    
    Parameters
    ----------
    dist: K.Tensor
        Tensor of shape (nb_features, nb_features, filters)
    k: Integer
        Number of top elements to retrieve

    Returns
    -------
    out: K.Tensors
        Tensor of shape (nb_features, nb_neighbors, filters)

    """
    # TODO: Update Documentation
    dist_shape = K.int_shape(dist)
    if K.backend() == 'tensorflow':
        swap_first_second_axes = [0, 2, 1]
        _, index = tf.nn.top_k(K.permute_dimensions(-dist, swap_first_second_axes), k=k)
        out = K.permute_dimensions(index, swap_first_second_axes)
    else:  # Theano
        index = T.argsort(dist, axis=1)
        out = index[:, :k, :]
    out._keras_shape = (dist_shape[1], k, dist_shape[2])
    return out


class PhyloConv1D(Conv1D):
    """1D phylo convolution layer

    This layer is divided in two logical step, first at every feature we add 
    the nb_neighbor - 1 closest features. After that we create a convolutional kernel 
    whom convolves the original feature with the added one in order to create a new
    metafeature which is then returned as output.
    
    # Arguments
        distances: K.Tensor, a tensor of shape (nb_features, nb_features)
            where distances[i, j] = the jth less distant feature from 
            the ith feature computed according to some algorithm. 
        nb_neighbors: Integer, the number of neighbors to be 
            added to any existing feature.
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        3D tensor with shape: `(batch_size, nb_features, nb_filters)`

    # Output shape
        3D tensor with shape: `(batch_size, nb_features, filters)`
    """
    def __init__(self,
                 distances,
                 nb_neighbors,
                 filters,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.nb_neighbors = nb_neighbors
        self.nb_features = K.int_shape(distances)[0]
        self.distances = distances

        super(PhyloConv1D, self).__init__(
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
            bias_constraint=bias_constraint,
            **kwargs
        )

        # TODO: Check if there is a better solution
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]

    def build(self, input_shape):
        """
        Since the build of Conv1D changes input_spec we need to 
        overwrite the change
        """
        input_spec = self.input_spec
        super(PhyloConv1D, self).build(input_shape[0])
        self.input_spec = input_spec

    def call(self, inputs, **kwargs):
        """
        
        Parameters
        ----------
        inputs: List of K.Tensors. The first entry is assumed to be
            the sample of our experiment while the second entry is 
            assumed to be the tensor of the coordinates associated with 
            the features that we are considering.

        Returns
        -------
        outputs: List of K.Tensor. The first entry is the convolution of the 
            sample after at every feature we added the nb_neighbors closest ones.
            The second entry is the convolution of the coordinates sample after 
            at every feature we added the nb_neighbors closest ones. Is important
            to notice that both convolutions use the same weights.

        """
        X = inputs[0]
        Coord = inputs[1]

        # Phylo neighbors step
        neighbor_indexes = _top_k(self.distances, k=self.nb_neighbors)
        X_phylongb = _gather_target_neighbors(X, neighbor_indexes)
        Coord_phylongb = _gather_target_neighbors(Coord, neighbor_indexes)

        # Convolution step
        X_conv = super(PhyloConv1D, self).call(X_phylongb)
        C_conv = super(PhyloConv1D, self).call(Coord_phylongb)

        outputs = [X_conv, C_conv]

        return outputs

    def compute_output_shape(self, input_shape):

        """
        In order to compute the correct output shape we need to consider
        that every feature is now followed by the nb_neighbors closest one
        """

        input_shape_X, input_shape_C = input_shape[0], input_shape[1]
        X_shape = (input_shape_X[0],
                   input_shape_X[1] * self.nb_neighbors,
                   input_shape_X[2])
        C_shape = (input_shape_C[0],
                   input_shape_C[1] * self.nb_neighbors,
                   input_shape_C[2])
        x = super(PhyloConv1D, self).compute_output_shape(X_shape)
        y = super(PhyloConv1D, self).compute_output_shape(C_shape)

        return [x, y]

    def compute_mask(self, inputs, mask=None):

        # TODO Add support for Masking, in case
        return [None, None]
