#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers import Layer, InputSpec
from keras.layers.convolutional import Conv1D, Conv2D

from keras import backend as K
import tensorflow as tf


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


def _euclidean_distances(X):
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
    distance *= -2
    distance += XX
    distance += YY
    distance = K.maximum(distance,
                         K.constant(0, dtype=distance.dtype))

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
    # TODO: FIX This function
    # TODO: Decide what to do with this stuff
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

    swap_first_second_axes = [0, 2, 1]

    _, index = tf.nn.top_k(K.permute_dimensions(dist, swap_first_second_axes), k=k)
    out = K.permute_dimensions(index, swap_first_second_axes)
    return out


class PhyloConv(Conv1D):
    """
    Wrapper for Conv1D that initializes the class with proper parameters.
    """

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
    """
    Keras layer that given an input of shape (batch_size, nb_features, filters) 
    returns a tensor of shape (batch_size, nb_features * nb_neighbors, filters) where every feature
    is followed by the nb_neighbors closest features. In this situation closest means closest according 
    to an euclidean metric computed on the R^n obtained applying an MDS algorithm to the features. Look at the 
    attached thesis "Phylogenetic Convolutional Neural Networks in Metagenomics" by Ylenia Giarratano for details.
    
    #Parameters:
        coordinates: K.Tensor of shape (nb_coordinates, nb_features, filters) 
            Tensor that for every filter k contains the MDS coordinates of the features contained in that filter.
        nb_neighbors: Integer
            Number of neighbors to be attached after every features
        nb_features: Integer
            Number of features to be considered
    """

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

    def call(self, inputs, **kwargs):
        """
        
        Parameters
        ----------
        inputs: K.Tensor of shape (batch_size, nb_features, filters)
            The tensor where we will add the neighbor to every feature

        Returns
        -------
        output:  K.Tensor of shape (batch_size, nb_features * nb_neighbor, filters)
            The tensor with all the neighbors added

        """

        neighbor_indexes = _top_k(-self.dist, k=self.nb_neighbors)

        # Add all the other features and their neighbors
        target_neighbors = neighbor_indexes[0: self.nb_features, 0:self.nb_neighbors, :]
        output = _gather_target_neighbors(inputs, target_neighbors)

        return output


def phylo_convutional_block(Xs, Crd, nb_neighbors, nb_features, filters):

    """
    Helper function that creates and execute a phylo-convolutional step on all the data
    
    Parameters
    ----------
    Xs: K.Tensor of shape (batch_size, nb_features, filters)
        Current data
        
    Crd: K.Tensor of shape (nb_coordinates, nb_features, filters)
        Current coordinates
    
    nb_neighbors: Integer
        number of neighbors to be considered in the convolutional step
    
    nb_features: Integer
        number of features to be considered in the convolutional step
    
    filters: Integer
        number of filters to be generated by the convolutional step

    Returns
    -------
    Xs_new:  K.Tensor of shape (batch_size, nb_features, filters)
        New data after convolution
    
    Crd_new: K.Tensor of shape (nb_coordinates, nb_features, filters)
        New coordinates after convolution
    """

    phngb = PhyloNeighbours(coordinates=Crd,
                            nb_neighbors=nb_neighbors,
                            nb_features=nb_features)
    phcnn = PhyloConv(nb_neighbors=nb_neighbors,
                      filters=filters)
    Xs_new = phcnn(phngb(Xs))
    Crd_new = phcnn(phngb(Crd))

    return Xs_new, Crd_new


class PhyloConv2(Conv2D):

    # TODO: Check if is really a conv1D what we want to do....

    def __init__(self,
                 coordinates,
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

        self.nb_features = coordinates.shape[1].value
        self.nb_neighbors = nb_neighbors
        self.dist = _euclidean_distances(coordinates)

        super(PhyloConv2, self).__init__(
            filters=filters,
            kernel_size=(1, nb_neighbors),
            strides=(1, nb_neighbors),
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

        self.input_spec = [InputSpec(shape=(None, coordinates.shape[1].value, coordinates.shape[2].value)),
                           InputSpec(shape=coordinates.shape)]

    def build(self, input_shape):

        conv_shape = (input_shape[0][0], 1) + (input_shape[0][1:])

        super(PhyloConv2, self).build(conv_shape)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]

    def _drop_second_dimension(self, X):

        second_axis_last = [0, 2, 3, 1]

        input_shape = [s.value if s.value else -1 for s in X.shape]
        output_shape = (input_shape[0], input_shape[2], input_shape[3])

        output = K.reshape(K.permute_dimensions(X, second_axis_last), output_shape)

        return output

    def call(self, inputs, **kwargs):

        X = inputs[0]
        Coord = inputs[1]

        # Phylo neighbors step

        neighbor_indexes = _top_k(-self.dist, k=self.nb_neighbors)
        target_neighbors = neighbor_indexes[0: self.nb_features, 0:self.nb_neighbors, :]
        X_phylongb = _gather_target_neighbors(X, target_neighbors)
        Coord_phylongb = _gather_target_neighbors(Coord, target_neighbors)

        # Convolution step

        X_conv = super(PhyloConv2, self).call(K.expand_dims(X_phylongb, 1))
        C_conv = super(PhyloConv2, self).call(K.expand_dims(Coord_phylongb, 1))

        X_conv = self._drop_second_dimension(X_conv)
        C_conv = self._drop_second_dimension(C_conv)

        output = [X_conv, C_conv]

        return output

    def compute_output_shape(self, input_shape):

        X_shape = (input_shape[0][0],
                   1,
                   input_shape[0][1] * self.nb_neighbors,
                   input_shape[0][2])
        C_shape = (input_shape[1][0],
                   1,
                   input_shape[1][1] * self.nb_neighbors,
                   input_shape[1][2])

        x = super(PhyloConv2, self).compute_output_shape(X_shape)
        y = super(PhyloConv2, self).compute_output_shape(C_shape)

        return [(x[0], x[2], x[3]),
                (y[0], y[2], y[3])]

    def compute_mask(self, inputs, mask=None):
        return [None, None]