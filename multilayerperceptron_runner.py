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
from dap.deep_learning_dap import DeepLearningDAP
from utils import get_data, to_list


class MultiLayerPerceptronDAP(DeepLearningDAP):
    """Specialisation of the DAP for DNN using Multi-Layer Perceptron model"""

    def __init__(self, experiment, target_disease):
        super(MultiLayerPerceptronDAP, self).__init__(experiment=experiment)
        self._disease_name = target_disease
        self.type_data = settings.TYPE_DATA
        self.total_nb_samples = settings.NB_SAMPLES
        self.hidden_units = settings.mlp_hidden_units
        self._do_serialisation = False # Serialization does not work with our Keras layer

    # ==== Abstract Methods Implementation ====

    @property
    def ml_model_name(self):
        return 'mlp'

    def _get_output_folder(self):
        """
        Compose path to the folder where reports and metrics will be saved.
        """
        base_filename = self._disease_name.lower()
        folder_name = '_'.join([base_filename, self.type_data, self.total_nb_samples, 
                                self.ml_model_name, str(self.hidden_units),
                                self.feature_scaling_name, self.feature_ranking_name,
                                str(self.cv_n), str(self.cv_k)])
        output_folder_path = os.path.join(settings.OUTPUT_DIR, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        return output_folder_path

    def _build_network(self):
        """
        Build a Multi-Layer Perceptron Network.
        
        Returns
        -------
        keras.models.Model
            MLP model
        """

        # Parameters for Input Layers
        nb_features = self._nb_features  # current nb of features in the feature step!

        # Parameter for output layer
        nb_classes = self.experiment_data.nb_classes
        hidden_units = self.hidden_units

        data = Input(shape=(nb_features, ), name="data", dtype=floatx())
        hidden = Dense(units=hidden_units, activation='sigmoid', name="hidden")(data)
        output = Dense(units=nb_classes, kernel_initializer="he_normal",
                       activation="softmax", name='output')(hidden)

        model = Model(inputs=data, outputs=output)
        return model

    def _save_extra_configuration(self):
        """Add extra steps to also store specific settings related to data file paths and directives."""
        settings_directives = dir(settings)
        settings_conf = {key: getattr(settings, key) for key in settings_directives if not key.startswith('__')}
        dump_filepath = os.path.join(self._get_output_folder(), 'mlp_settings.pickle')
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

    print('='*80)
    print('Processing {}'.format(datafile))
    print('='*80)

    dap = MultiLayerPerceptronDAP(inputs, settings.DISEASE)
    # dap.save_configuration()
    trained_model = dap.run(verbose=True)
    dap.predict_on_test(trained_model)

    # This is just because the TensorFlow version that we are using crashes
    # on completion. The message is just to be sure that the computation was terminated
    # before the segmentation fault
    print("Computation completed!")

if __name__ == '__main__':
    main()
