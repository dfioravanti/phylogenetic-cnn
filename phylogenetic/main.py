#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from . import settings
from utils import get_data, to_list
from data_analysis_plan import DeepLearningDAP

from keras.engine import Input, Model
from keras.layers import Lambda, MaxPool2D, Flatten, Dropout, Dense
from phcnn.layers import phcnn_conv_block


class PhyloDAP(DeepLearningDAP):
    """Specialisation of the DAP for DNN using PhyloCNN layers"""

    def __init__(self, experiment, target_disease):
        super(PhyloDAP, self).__init__(experiment=experiment)
        self._disease_name = target_disease
        self.nb_filters = to_list(settings.nb_convolutional_filters)
        self.phylo_neighbours = to_list(settings.nb_phylo_neighbours)

    @property
    def ml_model_name(self):
        return 'phylocnn'

    def _set_training_data(self):
        """Default implementation for classic and quite standard DAP implementation.
         More complex implementation require overriding this method.
        """
        super(PhyloDAP, self)._set_training_data()
        self.C = self.experiment_data.coordinates

    def _select_ranked_features(self, X_train, X_validation, ranked_feature_indices):
        """
        
        Parameters
        ----------
        X_train
        X_validation
        ranked_feature_indices

        Returns
        -------

        """
        self._C_fs = self.C[:,:,ranked_feature_indices]
        return super(PhyloDAP, self)._select_ranked_features(X_train, X_validation,
                                                             ranked_feature_indices)

    def _build_model(self):
        """Specific Implementation to instantiate the Deep Neural Network model."""

        filters = self.nb_filters
        nb_neighbours = self.phylo_neighbours
        nb_features = self.experiment_data.nb_features
        nb_coordinates = self.experiment_data.nb_coordinates
        nb_classes = self.experiment_data.nb_classes

        x = Input(shape=(1, nb_features, 1), name="data", dtype='float64')
        coordinates = Input(shape=(nb_coordinates, 1, nb_features, 1),
                            name="coordinates", dtype='float64')

        # TODO: Fix this one. Probably is better just throw an exception
        # if nb_neighbors > nb_features:
        #     nb_neighbors = nb_features

        conv = x
        conv_crd = Lambda(lambda c: c[0])(coordinates)

        for nb_filters, nb_neighbors in zip(filters, nb_neighbours):
            conv, conv_crd = phcnn_conv_block(conv, conv_crd, nb_neighbors,
                                              nb_features, nb_filters)

        max = MaxPool2D(pool_size=(1, 2), padding="valid")(conv)
        flatt = Flatten()(max)
        drop = Dropout(0, 1)(Dense(units=64)(flatt))
        output = Dense(units=nb_classes, kernel_initializer="he_normal",
                       activation="softmax", name='output')(drop)

        model = Model(inputs=[x, coordinates], outputs=output)

        opt = self.get_optimizer()
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

    def _fit_predict(self, model, X_train, y_train, X_validation, y_validation):
        """
        
        Parameters
        ----------
        model
        X_train
        y_train
        X_validation
        y_validation

        Returns
        -------

        """

        def _adjust_dimensions(Xs, Coord):
            return (np.expand_dims(np.expand_dims(Xs, 1), 3),
                    np.expand_dims(np.expand_dims(Coord, 2), 4))

        # Filter sample shapes on coordinates to match expected shapes in Input Tensors for
        # Train and Validation sets
        samples_in_train = X_train.shape[0]
        samples_in_val = X_validation.shape[0]
        C_train_fs = self._C_fs[:samples_in_train]
        C_val_fs = self._C_fs[:samples_in_val]

        # Adjust dimensions for Input Tensors
        X_train, C_train_fs = _adjust_dimensions(X_train, C_train_fs)
        X_val, C_val_fs = _adjust_dimensions(X_validation, C_val_fs)

        # Check Dimensions on Labels (must be categorical)


        model_history = model.fit({'data': X_train, 'coordinates': C_train_fs}, {'output': Ys_tr_cat},
                                  epochs=settings.epochs, verbose=settings.verbose,
                                  batch_size=settings.batch_size,
                                  validation_data=({'data': X_val,
                                                    'coordinates': C_val_fs}, {'output': Ys_val_cat}),
                                  )

        score, acc = model.evaluate({'data': X_val_fs, 'coordinates': Coords_val_sel},
                                    {'output': Ys_val_cat}, verbose=0)

        pv, p = self.predict_classes(model, [])


    def _get_output_folder_name(self):
        """
        :return: 
        """
        basefile_name = self._disease_name.lower()
        base_output_folder = os.path.join(settings.OUTPUT_DIR, '_'.join([basefile_name, self.ml_model_name,
                                                                         self.ranking_method.lower(),
                                                                         self.scaling_method.lower(),
                                                                         str(self.cv_n), str(self.cv_k)]))
        os.makedirs(base_output_folder, exist_ok=True)
        return base_output_folder

    def run(self, verbose=False):
        if len(self.nb_filters) != len(self.phylo_neighbours):
            raise Exception("nb_convolutional_filters and nb_phylo_neighbours must "
                            "have the same length. Check the config file")
        super(PhyloDAP, self).run(verbose)


def main():
    """"""
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    datafile = settings.TRAINING_DATA_FILEPATH
    labels_datafile = settings.TRAINING_LABELS_FILEPATH
    coordinates_datafile = settings.COORDINATES_FILEPATH
    test_datafile = settings.TEST_DATA_FILEPATH
    test_label_datafile = settings.TEST_LABELS_FILEPATH

    inputs = get_data(datafile, labels_datafile, coordinates_datafile,
                      test_datafile, test_label_datafile)

    dap_model = dap(inputs)

if __name__ == '__main__':
    main()

