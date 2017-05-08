import numpy as np
from sklearn.metrics import euclidean_distances


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