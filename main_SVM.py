#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn import svm

import settings
from dap import DAP

from utils import get_data


class SVM_DAP(DAP):
    """Specialisation of the DAP for SVM"""

    def __init__(self, experiment, target_disease):
        super(SVM_DAP, self).__init__(experiment=experiment)
        self._disease_name = target_disease
        self.type_data = settings.TYPE_DATA
        self.total_nb_samples = settings.NB_SAMPLES

        # SVM parameters for tuning

        self.C = 1.0
        self.tuning_pipeline = Pipeline([('scaler', self.feature_scaler),
                                         ('SVM', self._create_ml_model())
                                        ])
    # ==== Abstract Methods Implementation ====

    @property
    def ml_model_name(self):
        return 'SVM'

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

    def _create_ml_model(self):
        """Instantiate a new Machine Learning model to be used in the fit-predict step.
        Most likely this function has to simply call the constructor of the
        `sklearn.Estimator` object to be used.

        Examples:
        ---------
        """

        return svm.SVC(C=self.C, kernel=settings.kernel)

    def _extra_operations_begin_fold(self):

        print("-------------------------")
        print("Tuning SVM")
        print("-------------------------")

        Xs_tr = self.X[self._fold_training_indices]
        ys_tr = self.y[self._fold_training_indices]

        tuning_cross_validation = model_selection.StratifiedShuffleSplit(n_splits=settings.tuning_cv_K,
                                                                         test_size=settings.tuning_cv_P,
                                                                         random_state=self._fold_nb)

        grid_search = model_selection.GridSearchCV(estimator=self.tuning_pipeline,
                                                   param_grid=settings.tuning_parameter,
                                                   cv=tuning_cross_validation,
                                                   scoring=settings.scorer)

        grid_search.fit(Xs_tr, ys_tr)
        self.C = grid_search.best_estimator_.get_params()['SVM__C']

    def _train_best_model(self, k_feature_indices, seed=None):
        self._extra_operations_begin_fold()
        return super(SVM_DAP, self)._train_best_model(k_feature_indices, seed)


def main():

    datafile = settings.TRAINING_DATA_FILEPATH
    labels_datafile = settings.TRAINING_LABELS_FILEPATH
    coordinates_datafile = settings.COORDINATES_FILEPATH
    test_datafile = settings.TEST_DATA_FILEPATH
    test_label_datafile = settings.TEST_LABELS_FILEPATH

    inputs = get_data(datafile, labels_datafile, coordinates_datafile,
                      test_datafile, test_label_datafile)

    dap = SVM_DAP(inputs, settings.DISEASE)
    # dap.save_configuration()
    trained_model = dap.run(verbose=True)
    dap.predict_on_test(trained_model)

    # This is just because the TensorFlow version that we are using crashes
    # on completion. The message is just to be sure that the computation was terminated
    # before the segmentation fault
    print("Computation completed!")

if __name__ == '__main__':
    main()
