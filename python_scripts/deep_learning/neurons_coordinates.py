from __future__ import division
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
'''This function prepare the input for the next model:
    The input has to have shape (1, nb_filter, nrow, ncol)*k_2
    where k_2 is the number of neighbors that we want to keep in cosideration.
    Since depending on the filter the position of the neighbors changes
    because we do a combination between filter and coordinates
    we have to be careful to add the right neighbors for each level of the first output'''


def neurons_coordinates(layer_conv_output,layer_max_output, coordinates_red, w, nb_filter, k_1, k_2):
    'Reordering as in the first layer'
    coordinate_adjoint1 = np.zeros((coordinates_red.shape[0],coordinates_red.shape[1]*k_1)) #(258,259) # coordinates in columns
    dist = euclidean_distances(coordinates_red.transpose()) #transpose because use distance between rows
    neighbors = np.zeros((coordinates_red.shape[1],coordinates_red.shape[1]), dtype='int')
    W = np.zeros((layer_max_output.shape[0],1,1,layer_max_output.shape[3]*k_2))
    W = np.empty_like(W)
    Z = np.zeros((1,1,1,layer_max_output.shape[3]*k_2))
    Z = np.empty_like(Z)
    #print layer_max_output.shape[0]
    #print layer_conv_output.shape[0]
    #print 'Z', np.shape(Z)
    #print 'W', np.shape(W)
    for l in range(coordinates_red.shape[1]):
        neighbors[l] = np.argsort(dist[l])

    for i in range(coordinates_red.shape[0]):
        for col in range(coordinates_red.shape[1]*k_1):
            res=int((col%k_1))
            j = int((col/k_1))
            coordinate_adjoint1[i,col] = coordinates_red[i, neighbors[j,res]]
    '''At this point we have the k_1-neighbors next to each variable
    we have to reproduce the first convolution for the coordinates'''

    for filter in range(nb_filter):
        'Convolve the old coordinate (without bias)'
        coordinate_conv1= np.ndarray(shape=(coordinates_red.shape[0],coordinates_red.shape[1])) # coordinates of the new neurons
        for list in range(np.shape(coordinate_conv1)[0]):
            #print list, filter
            for elem in range(0,np.shape(coordinate_adjoint1)[1],k_1):
                coordinate_conv1[list][int(elem/k_1)]=np.convolve(coordinate_adjoint1[list][elem:elem+k_1],w[filter],mode='valid')
            #print 'np.shape(layer_conv_output)', np.shape(layer_conv_output)
        for sample in range(layer_conv_output.shape[0]):
            layer_conv_output_1 = layer_conv_output[sample,filter,0,:] #(259,)
            #print 'np.shape(layer_conv_output_1)', np.shape(layer_conv_output_1)
            layer_max_output_1 = layer_max_output[sample,filter,0,:]
            index_max_output = []
            for num in range(0,len(layer_conv_output_1), 2):
                index_max_output.append(np.argmax(layer_conv_output_1[num:num+2])+num)
                #print layer_conv_output_1[num:num+2]
            #print "index", index_max_output
            #print len(layer_max_output_1)

            z = phyloneighbors(x=np.reshape(layer_max_output_1,(1,len(layer_max_output_1))), coordinates= coordinate_conv1[:,index_max_output],k=k_2)
            #print 'np.shape', np.shape(z)
            if sample==0:
                Z=z
            else:
                Z = np.concatenate((Z,z), axis=0)
                #print 'np.shape(Z)', 'sample', np.shape(Z), sample
        #print np.shape(Z)
        if filter==0:
            W = Z
        else:
            W = np.concatenate((W,Z), axis=1)
        #print 'np.shape(W)', np.shape(W), nb_filter

    #print 'np.shape(W)', np.shape(W), nb_filter
    return W


def phyloneighbors(x, coordinates,k):
    dist = euclidean_distances(coordinates.transpose())
    #print np.shape(dist)

    neighbors = np.zeros((coordinates.shape[1],coordinates.shape[1]), dtype='int')
    for l in range(coordinates.shape[1]):
        neighbors[l] = np.argsort(dist[l])
    #print np.shape(neighbors)
    output = np.zeros((x.shape[0],x.shape[1]*k))
    for col in range(x.shape[1]*k):
            #print 'sono dento phylone',x.shape[1]*k
            res=int((col%k))
            j = int((col/k))
            #print 'col:', col,'res:' ,res,'j:', j, 'k:',k, 'col%k', col%k, int((col%k))
            output[0,col] =  x[0,neighbors[j,res]]
    output = np.reshape(output, (output.shape[0],1,1,output.shape[1]))
    return output