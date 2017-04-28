from __future__ import division
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Reshape, Flatten
from sklearn.metrics.pairwise import euclidean_distances
from keras.optimizers import SGD, Adam
from keras.models import Model
import performance as perf
import keras.engine
import numpy as np
import csv
import mlpy
from input_output import load_data

from coordinate_1dembes import PhyloNeighbors
np.random.seed(2016)
from keras import backend as K
import os
import sys
sys.setrecursionlimit(20000)

home_path = '/Users/yleniagiarratano'

'------------------------------------------LOAD THE DATA------------------------------------------------------------------------'
x_data = np.genfromtxt(home_path + '/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training.txt')[1:,1:]
print 'x_data', np.shape(x_data)
x_valid = np.genfromtxt(home_path + '/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_validation.txt')[1:,1:]

target  =  np.genfromtxt(home_path + '/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_training_lab.txt', "float32")
target_valid  =  np.genfromtxt(home_path + '/Desktop/ThesisData/progettoR/HS_CDf/Sokol_16S_taxa_HS_CDflare_commsamp_validation_lab.txt', "float32")
y  =  [ [0,1] if a==1 else [1,0] for a in target]
_,_,coordinates = load_data(home_path + '/Desktop/ThesisData/progettoR/coordinates_cdf.txt') #
coordinates = np.asarray(coordinates, dtype="float32") # coordinates are in columns
#print x_valid.shape
#print x_data.shape
x =  x_data
x_data = np.reshape(x_data,(x_data.shape[0],1,1,x_data.shape[1]))
print 'x_data', np.shape(x_data)
print 'x.shape', x.shape[1]
x_valid = np.reshape(x_valid, (x_valid.shape[0],1,1,x_valid.shape[1]))
#coordinate = np.reshape(coordinates,(1,1,coordinates.shape[0],coordinates.shape[1]))
k=3
'---------------------------------------BUILD THE FIRST NETWORK-------------------------------------------------------------'
# Model1 for the first abundances
model = Sequential()
model.add(PhyloNeighbors(input_shape=(1,1,x.shape[1]), k=k, coordinates=coordinates, output_dim=x_data.shape[3]*k)) # mettere no trainable
model.add(Convolution2D( nb_filter=1, nb_row=1, nb_col=k, border_mode='valid', subsample=(1,k)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss= "binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()
model.fit(x_data,y, nb_epoch=1, batch_size=1)
p = model.predict_classes(x_data, batch_size=1, verbose=0)
p_valid = model.predict_classes(x_valid, batch_size=1, verbose=0)

NPV = perf.npv(target, p)
PPV = perf.ppv(target, p)
SENS = perf.sensitivity(target, p)
SPEC = perf.specificity(target, p)
MCC = perf.KCCC_discrete(target, p)
print "MCC in train", MCC

print "predicted_label_valid", p_valid
print "real_label_valid", target_valid
NPV = perf.npv(target_valid, p_valid)
PPV = perf.ppv(target_valid, p_valid)
SENS = perf.sensitivity(target_valid, p_valid)
SPEC = perf.specificity(target_valid, p_valid)
MCC = perf.KCCC_discrete(target_valid, p_valid)
print "MCC in validation", MCC

'-------------------------------------------SECOND PART-------------------------------------------------------'
'------------------------------------------FIX COORDINATES----------------------------------------------------'
'New coordinate after convolution'
get_conv_layer_output = K.function([model.layers[0].input],[model.layers[1].output])
layer_output = get_conv_layer_output([x_data])[0]

conv1 = model.layers[1]
weights_conv1 = conv1.get_weights()
w = weights_conv1[0] # it is 4D (n_filters, 1,1, k)
bias =  weights_conv1[1]
w = np.reshape(w,(w.shape[0],w.shape[3]))

