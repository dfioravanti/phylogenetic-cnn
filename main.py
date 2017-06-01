#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from keras.backend import floatx
from keras.engine import Input, Model
from keras.layers import (Lambda, MaxPooling1D, Flatten,
                          Dropout, Dense, BatchNormalization)

import pickle
import settings
from dap import DeepLearningDAP
from phcnn.layers import PhyloConv1D, euclidean_distances
from utils import get_data, to_list


class PhyloDAP(DeepLearningDAP):
    """Specialisation of the DAP for DNN using PhyloCNN layers"""

    def __init__(self, experiment, target_disease):
        super(PhyloDAP, self).__init__(experiment=experiment)
        self._disease_name = target_disease
        self.type_data = settings.TYPE_DATA
        self.total_nb_samples = settings.NB_SAMPLES
        self.nb_filters = to_list(settings.nb_convolutional_filters)
        self.phylo_neighbours = to_list(settings.nb_phylo_neighbours)

        self._do_serialisation = False # Serialization does not work with our Keras layer

    # ==== Abstract Methods Implementation ====

    @property
    def ml_model_name(self):
        return 'phylocnn'

    def _get_output_folder(self):
        """
        Compose path to the folder where reports and metrics will be saved.
        """
        base_filename = self._disease_name.lower()
        folder_name = '_'.join([base_filename, self.type_data, self.total_nb_samples, self.ml_model_name,
                                self.feature_scaling_name, self.feature_ranking_name,
                                str(self.cv_n), str(self.cv_k)])
        output_folder_path = os.path.join(settings.OUTPUT_DIR, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        return output_folder_path

    def _build_network(self):
        """
        Build a PhyloCNN Network.
        
        Returns
        -------
        keras.models.Model
            PhyloCNN model
        """

        # Parameters for Input Layers
        nb_features = self._nb_features  # current nb of features in the feature step!
        nb_coordinates = self.experiment_data.nb_coordinates

        # Paramenters for phylo_conv layers
        list_filters = self.nb_filters
        list_neighbours = self.phylo_neighbours

        # Parameter for output layer
        nb_classes = self.experiment_data.nb_classes

        data = Input(shape=(nb_features, 1), name="data", dtype=floatx())
        coordinates = Input(shape=(nb_coordinates, nb_features, 1),
                            name="coordinates", dtype=floatx())

        conv_layer = data

        # We remove the padding that we added to work around keras limitations
        conv_crd = Lambda(lambda c: c[0], output_shape=lambda s: (s[1:]))(coordinates)

        for nb_filters, nb_neighbors in zip(self.nb_filters, self.phylo_neighbours):

            if nb_neighbors > nb_features:
                raise Exception("More neighbors than features, " \
                                "please use less neighbors or use more features")

            distances = euclidean_distances(conv_crd)
            conv_layer, conv_crd = PhyloConv1D(distances, nb_neighbors, nb_filters)([conv_layer, conv_crd])

            conv_layer = BatchNormalization(axis=1)(conv_layer)
            conv_layer = Dropout(0.25, seed=np.random.seed())(conv_layer)

        max = MaxPooling1D(pool_size=2, padding="valid")(conv_layer)
        flatt = Flatten()(max)
        drop = Dropout(0.25)(Dense(units=64)(flatt))
        output = Dense(units=nb_classes, kernel_initializer="he_normal",
                       activation="softmax", name='output')(drop)

        model = Model(inputs=[data, coordinates], outputs=output)
        return model

    @staticmethod
    def custom_layers_objects():
        """Return the dictionary mapping PhyloConv1D layers to 
        proper objects for correct de-serialisation of cached models."""
        return {'PhyloConv1D': PhyloConv1D}

    # ==== Overriding (some of) default methods behaviours ====

    def _set_training_data(self):
        """
        Add set of coordinates to default training data
        """
        super(PhyloDAP, self)._set_training_data()
        self.C = self.experiment_data.coordinates

    def _set_test_data(self):
        super(PhyloDAP, self)._set_test_data()
        self.C = self.experiment_data.coordinates

    def _select_ranked_features(self, ranked_feature_indices, X_train, X_validation=None):
        """
        Apply ranking filter also on Coordinates, 
        in addition to default filtering on training and validation data.
          
        Parameters
        ----------
        X_train: array-like, shape = [n_samples, n_features]
            Training data
        
        X_validation: array-like, shape = [n_samples, n_features]
            Validation data
            
        ranked_feature_indices: X_train: array-like, shape = [n_features_step]
            Array of indices corresponding to features to select.
        
        Returns
        -------
        X_train: array-like, shape = [n_samples, n_features_step]
            Training data with selected features
        
        X_validation: array-like, shape = [n_samples, n_features_step]
            Validation data with selected features
        """

        # Store new attributes referring to set of coordinates in feature steps.
        # i.e. `self._C_fs`
        self._C_fs_train = self.C[..., ranked_feature_indices]
        if X_validation is not None:
            self._C_fs_val = self.C[..., ranked_feature_indices]
        return super(PhyloDAP, self)._select_ranked_features(ranked_feature_indices, X_train, X_validation)

    @staticmethod
    def _adjust_dimensions(X, Coord):
        """
        Utility method used for input data preparation.
        """
        return np.expand_dims(X, 3), np.expand_dims(Coord, 4)

    def _prepare_data(self, X, training_data=True):
        """
        Implement Data Preparation before training/inference steps.
        Data preparation involves adjusting dimensions to 
        input data and corresponding coordinates.
        
        Adjusted data will be returned as a Python list - supported
        by Keras models.
        
        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features)
            Input data to prepare
            
        training_data: bool (default: True)
            Flag indicating whether input data are training data or not.
            This flag is included as it may be required to prepare data
            differently depending they're training or validation data.

        Returns
        -------
        data: list
            List of data and coordinates 
        """

        # Filter sample shapes on coordinates to match expected shapes
        # in Input Tensors

        samples_in_batch = X.shape[0]
        if training_data:
            C_fs = self._C_fs_train[:samples_in_batch]
        else:
            C_fs = self._C_fs_val[:samples_in_batch]
        X, C_fs = self._adjust_dimensions(X, C_fs)
        return [X, C_fs]

    def run(self, verbose=False):
        if len(self.nb_filters) != len(self.phylo_neighbours):
            raise Exception("nb_convolutional_filters and nb_phylo_neighbours must "
                            "have the same length. Check the config file")
        return super(PhyloDAP, self).run(verbose)

    def _save_extra_configuration(self):
        """Add extra steps to also store specific settings related to data file paths and directives."""
        settings_directives = dir(settings)
        settings_conf = {key: getattr(settings, key) for key in settings_directives if not key.startswith('__')}
        dump_filepath = os.path.join(self._get_output_folder(), 'phylodap_settings.pickle')
        with open(dump_filepath, "wb") as dump_file:
            pickle.dump(obj=settings_conf, file=dump_file, protocol=pickle.HIGHEST_PROTOCOL)


def main():

    datafile = settings.TRAINING_DATA_FILEPATH
    labels_datafile = settings.TRAINING_LABELS_FILEPATH
    coordinates_datafile = settings.COORDINATES_FILEPATH
    test_datafile = settings.TEST_DATA_FILEPATH
    test_label_datafile = settings.TEST_LABELS_FILEPATH

    inputs = get_data(datafile, labels_datafile, coordinates_datafile,
                      test_datafile, test_label_datafile)

    dap = PhyloDAP(inputs, settings.DISEASE)
    # dap.save_configuration()
    trained_model = dap.run(verbose=True)
    dap.predict_on_test(trained_model)

    # This is just because the TensorFlow version that we are using crashes
    # on completion. The message is just to be sure that the computation was terminated
    # before the segmentation fault
    print("Computation completed!")

if __name__ == '__main__':
    main()
