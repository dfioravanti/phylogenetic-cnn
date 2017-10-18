#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
from keras.backend import floatx
from keras.engine import Input, Model
from keras.layers import (Lambda, MaxPooling1D, Flatten,
                          Dropout, Dense)

import pickle
import settings
from dap.deep_learning_dap import DeepLearningDAP
from phcnn.layers import PhyloConv1D, euclidean_distances
from utils import get_data, to_list

from keras import backend as K


class PhyloDAP(DeepLearningDAP):
    """Specialisation of the DAP for DNN using PhyloCNN layers"""

    def __init__(self, experiment, dbname):
        super(PhyloDAP, self).__init__(experiment=experiment)
        self.db_name = dbname
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
        base_filename = self.db_name.lower()
        folder_name = '_'.join([base_filename, self.ml_model_name,
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

        def coords(X):
            return [X, K.in_train_phase(K.variable(self._C_fs_train, dtype=K.floatx()),
                                        K.variable(self._C_fs_test, dtype=K.floatx()))]

        # Parameters for Input Layers
        nb_features = self._nb_features  # current nb of features in the feature step!

        #  Paramenters for phylo_conv layers

        # Parameter for output layer
        nb_classes = self.experiment_data.nb_classes

        data = Input(shape=(nb_features, 1), name="data", dtype=floatx())
        # We remove the padding that we added to work around keras limitations
        x, conv_crd = Lambda(coords)(data)

        conv_layer = x
        for nb_filters, nb_neighbors in zip(self.nb_filters, self.phylo_neighbours):

            if nb_neighbors > nb_features:
                raise Exception("More neighbors than features, "
                                "please use less neighbors or use more features")

            distances = euclidean_distances(conv_crd)
            conv_layer, conv_crd = PhyloConv1D(distances,
                                               nb_neighbors,
                                               nb_filters, activation='selu')([conv_layer, conv_crd])

        max = MaxPooling1D(pool_size=2, padding="valid")(conv_layer)
        flatt = Flatten()(max)
        drop = Dropout(0.25)(Dense(units=64, activation='selu')(flatt))
        output = Dense(units=nb_classes, kernel_initializer="he_normal",
                       activation="softmax", name='output')(drop)

        model = Model(inputs=data, outputs=output)
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
        self.C_train = self.experiment_data.coordinates
        self.C_train = np.expand_dims(self.C_train, 3)

    def _set_test_data(self):
        super(PhyloDAP, self)._set_test_data()
        self.C_test = self.experiment_data.test_coordinates
        self.C_test = np.expand_dims(self.C_test, 3)

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
        self._C_fs_train = self.C_train[:, ranked_feature_indices, :]
        self._C_fs_test = self.C_test[:, ranked_feature_indices, :]
        return super(PhyloDAP, self)._select_ranked_features(ranked_feature_indices, X_train, X_validation)

    # def predict_on_test(self, best_model):
    #     """
    #     Execute the last step of the DAP. A prediction using the best model
    #     trained in the main loop and the best number of features.
    #
    #     Parameters
    #     ----------
    #     best_model
    #         The best model trained by the run() method
    #     """
    #
    #     self._set_test_data()
    #     X_test = self.X_test
    #     Y_test = self.y_test
    #
    #     if self.apply_feature_scaling:
    #         _, X_test = self._apply_scaling(self.X, self.X_test)
    #
    #     # Select the correct features and prepare the data before predict
    #     feature_ranking = self._best_feature_ranking[:self._nb_features]
    #     X_test = self._select_ranked_features(feature_ranking, X_test)
    #     X_test = self._prepare_data(X_test, learning_phase=False)
    #     Y_test = self._prepare_targets(Y_test)
    #
    #     predictions = self._predict(best_model, X_test)
    #     self._compute_test_metrics(Y_test, predictions)
    #     self._save_test_metrics_to_file(self._get_output_folder())

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
        X = np.expand_dims(X, 3)
        return X

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
    coordinates_datafile = settings.TRAINING_COORDINATES_FILEPATH
    test_datafile = settings.TEST_DATA_FILEPATH
    test_label_datafile = settings.TEST_LABELS_FILEPATH
    test_coordinates_datafile = settings.TEST_COORDINATES_FILEPATH

    inputs = get_data(datafile, labels_datafile, coordinates_datafile,
                      test_datafile, test_label_datafile, test_coordinates_datafile)

    dap = PhyloDAP(inputs, settings.DATABASE_NAME)
    # dap.save_configuration()
    trained_model = dap.run(verbose=True)
    dap.predict_on_test(trained_model)

    # This is just because the TensorFlow version that we are using crashes
    # on completion. The message is just to be sure that the computation was terminated
    # before the segmentation fault
    print("Computation completed!")

if __name__ == '__main__':
    main()
