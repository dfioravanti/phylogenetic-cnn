"""Main module implementing Data Analysis Plan
"""

import os
import mlpy
import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.models import model_from_json

from . import settings
from .scaling import get_feature_scaling_name
from .ranking import get_feature_ranking_name
from .performance import (npv, ppv, sensitivity, specificity,
                          KCCC_discrete, dor, accuracy)
from sklearn.metrics import roc_auc_score

from keras.callbacks import ModelCheckpoint

from abc import ABC, abstractmethod


class DAP(ABC):
    """Data Analysis Plan ABC"""

    # Metrics Keys
    ACC = 'ACC'
    DOR = 'DOR'
    AUC = 'AUC'
    MCC = 'MCC'
    SPEC = 'SPEC'
    SENS = 'SENS'
    PPV = 'PPV'
    NPV = 'NPV'
    RANKINGS = 'ranking'
    PREDS = 'PREDS'

    BASE_METRICS = [ACC, DOR, AUC, MCC, SPEC, SENS, PPV, NPV, RANKINGS, PREDS]

    # Confidence Intervals
    MCC_CI = 'MCC_CI'
    SPEC_CI = 'SPEC_CI'
    SENS_CI = 'SENS_CI'
    PPV_CI = 'PPV_CI'
    NPV_CI = 'NPV_CI'
    ACC_CI = 'ACC_CI'
    DOR_CI = 'DOR_CI'
    AUC_CI = 'AUC_CI'

    CI_METRICS = [MCC_CI, SPEC_CI, SENS_CI, DOR_CI, ACC_CI, AUC_CI, PPV_CI, NPV_CI]

    TEST = 'TEST'

    def __init__(self, experiment):
        """
        
        Parameters
        ----------
        experiment: sklearn.dataset.base.Bunch
            A bunch (dictionary-like) objects embedding 
            all data and corresponding attributes related 
            to the current experiment.
            
        """
        self.experiment_data = experiment

        # Map DAP configurations from settings to class attributes
        self.cv_k = settings.Cv_K
        self.cv_n = settings.Cv_N
        self.feature_ranker = settings.feature_ranker
        self.feature_scaler = settings.feature_scaler
        self.random_labels = settings.use_random_labels
        self.is_stratified = settings.stratified
        self.to_categorical = settings.to_categorical
        self.apply_feature_scaling = settings.apply_feature_scaling

        self.iteration_steps = self.cv_k * self.cv_n
        self.feature_steps = len(settings.feature_ranges)
        if settings.use_top_feature:
            self.feature_steps += 1

        # Initialise Metrics Arrays
        self.metrics = self._prepare_metrics_array()

        # Set Training Dataset
        self._set_training_data()

        if self.random_labels:
            np.random.shuffle(self.y)

        # Model is forced to None.
        # The `create_ml_model` abstract method must be implemented to return a new
        # ML model to be used in the `fit_predict` step.
        self.ml_model_ = None

        # Store reference to the best feature ranking
        self._best_feature_ranking = None

        # Contextual Information:
        # -----------------------
        # Attributes saving information on the context of the DAP process,
        # namely the reference to the iteration no. of the CV,
        # corresponding feature step, and so on.
        # This information will be updated throughout the execution of the process,
        # keeping track of current actual progresses.
        self._fold_training_indices = None  # indices of samples in training set
        self._fold_validation_indices = None  # indices of samples in validation set
        self._iteration_step_nb = -1
        self._feature_step_nb = -1
        self._runstep_nb = -1
        self._fold_nb = -1
        # Store the number of features in each iteration/feature step.
        self._nb_features = -1
        # Note: This attribute is not really used in this general DAP process,
        # although this is paramount for its DeepLearning extension.
        # In fact it is mandatory to know
        # the total number of features to be used so to properly
        # set the shape of the first InputLayer(s).
        # Contextual information exposed in order to be used for verbose printing

    @property
    def ml_model(self):
        """Machine Learning Model to be used in DAP."""
        if not self.ml_model_:
            self.ml_model_ = self._create_ml_model()
        return self.ml_model_

    @property
    def feature_scaling_name(self):
        if isinstance(self.feature_scaler, str):
            return self.feature_scaler
        return get_feature_scaling_name(self.feature_scaler)

    @property
    def feature_ranking_name(self):
        if isinstance(self.feature_ranker, str):
            return self.feature_ranker
        return get_feature_ranking_name(self.feature_ranker)

    # ====== Abstract Methods ======
    #
    @abstractmethod
    def _create_ml_model(self):
        """Instantiate a new Machine Learning model to be used in the fit-predict step.
        Most likely this function has to simply call the constructor of the 
        `sklearn.Estimator` object to be used.
        
        Examples:
        ---------
        
        from sklearn.svm import SVC
        return SVC(kernel='rbf', C=0.01)
        """

    @abstractmethod
    def _get_output_folder(self):
        """Return the path to the output folder. 
        This method is abstract as its implementation is very experiment-dependent!
        
        Returns
        -------
        str : Path to the output folder
        """

    @property
    @abstractmethod
    def ml_model_name(self):
        """Abstract property for machine learning model associated to DAP instance.
        Each subclass should implement this property, if needed."""

    # ====== Utility Methods =======
    #
    def _set_training_data(self):
        """Default implementation for classic and quite standard DAP implementation.
         More complex implementation require overriding this method.
        """
        self.X = self.experiment_data.training_data
        self.y = self.experiment_data.targets

    def _prepare_metrics_array(self):
        """
        Initialise Base "Standard" DAP Metrics to be monitored and saved during the 
        data analysis plan.
        """

        metrics_shape = (self.iteration_steps, self.feature_steps)
        metrics = {
            self.RANKINGS: np.empty((self.iteration_steps, self.experiment_data.nb_features), dtype=np.int),
            self.NPV: np.empty(metrics_shape),
            self.PPV: np.empty(metrics_shape),
            self.SENS: np.empty(metrics_shape),
            self.SPEC: np.empty(metrics_shape),
            self.MCC: np.empty(metrics_shape),
            self.AUC: np.empty(metrics_shape),
            self.ACC: np.empty(metrics_shape),
            self.DOR: np.empty(metrics_shape),
            self.PREDS: np.empty(metrics_shape + (self.experiment_data.nb_samples,), dtype=np.int),

            # Confidence Interval Metrics-specific
            self.MCC_CI: np.empty((self.feature_steps, 3)),
            self.ACC_CI: np.empty((self.feature_steps, 3)),
            self.AUC_CI: np.empty((self.feature_steps, 3)),
            self.PPV_CI: np.empty((self.feature_steps, 3)),
            self.NPV_CI: np.empty((self.feature_steps, 3)),
            self.SENS_CI: np.empty((self.feature_steps, 3)),
            self.SPEC_CI: np.empty((self.feature_steps, 3)),
            self.DOR_CI: np.empty((self.feature_steps, 3)),

            # Test dictionary

            self.TEST: dict()
        }
        # Initialize to Flag Values
        metrics[self.PREDS][:, :, :] = -10
        return metrics

    def _compute_step_metrics(self, validation_indices, validation_labels,
                              predicted_labels, predicted_class_probs, **extra_metrics):
        """
        Compute the "classic" DAP Step metrics for corresponding iteration-step and feature-step.
        
        Parameters
        ----------
        validation_indices: array-like, shape = [n_samples]
            Indices of validation samples
            
        validation_labels: array-like, shape = [n_samples]
            Array of labels for samples in the validation set
            
        predicted_labels: array-like, shape = [n_samples]
            Array of predicted labels for each sample in the validation set
            
        predicted_class_probs: array-like, shape = [n_samples, n_classes]
            matrix containing the probability for each class and for each sample in the
            validation set
            
        Other Parameters
        ----------------
            
        extra_metrics:
            List of extra metrics to log during execution returned by the `_fit_predict` method
            This list will be processed separately from standard "base" metrics.
        """

        # Compute Base Step Metrics
        iteration_step, feature_step = self._iteration_step_nb, self._feature_step_nb
        self.metrics[self.PREDS][iteration_step, feature_step, validation_indices] = predicted_labels
        self.metrics[self.NPV][iteration_step, feature_step] = npv(validation_labels, predicted_labels)
        self.metrics[self.PPV][iteration_step, feature_step] = ppv(validation_labels, predicted_labels)
        self.metrics[self.SENS][iteration_step, feature_step] = sensitivity(validation_labels, predicted_labels)
        self.metrics[self.SPEC][iteration_step, feature_step] = specificity(validation_labels, predicted_labels)
        self.metrics[self.MCC][iteration_step, feature_step] = KCCC_discrete(validation_labels, predicted_labels)
        self.metrics[self.AUC][iteration_step, feature_step] = roc_auc_score(validation_labels, predicted_labels)
        self.metrics[self.DOR][iteration_step, feature_step] = dor(validation_labels, predicted_labels)
        self.metrics[self.ACC][iteration_step, feature_step] = accuracy(validation_labels, predicted_labels)

        if extra_metrics:
            self._compute_extra_step_metrics(validation_indices, validation_labels,
                                             predicted_labels, predicted_class_probs, **extra_metrics)

    def _compute_extra_step_metrics(self, validation_indices=None, validation_labels=None,
                                    predicted_labels=None, predicted_class_probs=None, **extra_metrics):
        """Method to be implemented in case extra metrics are returned during the fit-predict step.
           By default, no additional extra metrics are returned.
           
           Parameters are all the same of the "default" _compute_step_metrics method, with the 
           exception that can be None to make the API even more flexible!
        """
        pass

    def _compute_test_metrics(self, test_labels, predicted_labels,
                              predicted_class_probs, **extra_metrics):

        """
        Compute the "classic" DAP Step metrics for test data

        Parameters
        ----------

        test_labels: array-like, shape = [n_samples]
            Array of labels for samples in the validation set

        predicted_labels: array-like, shape = [n_samples]
            Array of predicted labels for each sample in the validation set

        predicted_class_probs: array-like, shape = [n_samples, n_classes]
            matrix containing the probability for each class and for each sample in the
            validation set

        Other Parameters
        ----------------

        extra_metrics:
            List of extra metrics to log during execution returned by the `predict` method
            This list will be processed separately from standard "base" metrics.
        """

        self.metrics[self.TEST][self.PREDS] = predicted_labels
        self.metrics[self.TEST][self.NPV] = npv(test_labels, predicted_labels)
        self.metrics[self.TEST][self.PPV] = ppv(test_labels, predicted_labels)
        self.metrics[self.TEST][self.SENS] = sensitivity(test_labels, predicted_labels)
        self.metrics[self.TEST][self.SPEC] = specificity(test_labels, predicted_labels)
        self.metrics[self.TEST][self.MCC] = KCCC_discrete(test_labels, predicted_labels)
        self.metrics[self.TEST][self.AUC] = roc_auc_score(test_labels, predicted_labels)
        self.metrics[self.TEST][self.DOR] = dor(test_labels, predicted_labels)
        self.metrics[self.TEST][self.ACC] = accuracy(test_labels, predicted_labels)

        if extra_metrics:
            self._compute_extra_test_metrics(test_labels, predicted_labels,
                                             predicted_class_probs, **extra_metrics)

    def _compute_extra_test_metrics(self, test_labels=None, predicted_labels=None,
                                    predicted_class_probs=None, **extra_metrics):
        """Method to be implemented in case extra metrics are returned during the predict step.
           By default, no additional extra metrics are returned.

           Parameters are all the same of the "default" _compute_test_metrics method, with the 
           exception that can be None to make the API even more flexible!
        """
        pass

    def _compute_metrics_confidence_intervals(self, feature_steps):
        """Compute Confidence Intervals for all target metrics.
        
        Parameters
        ----------
        feature_steps: list
            List of feature steps considered during the DAP process
        """

        # Compute Confidence Intervals for Metrics

        def _ci_metric(ci_metric_key, metric_key):
            metric_means = np.mean(self.metrics[metric_key], axis=0)
            for step, _ in enumerate(feature_steps):
                metric_mean = metric_means[step]
                ci_low, ci_hi = mlpy.bootstrap_ci(self.metrics[metric_key][:, step])
                self.metrics[ci_metric_key][step] = np.array([metric_mean, ci_low, ci_hi])

        _ci_metric(self.MCC_CI, self.MCC)  # MCC
        _ci_metric(self.ACC_CI, self.ACC)  # ACC
        _ci_metric(self.AUC_CI, self.AUC)  # AUC
        _ci_metric(self.DOR_CI, self.DOR)  # DOR
        _ci_metric(self.PPV_CI, self.PPV)  # PPV
        _ci_metric(self.NPV_CI, self.NPV)  # NPV
        _ci_metric(self.SPEC_CI, self.SPEC)  # SPEC
        _ci_metric(self.SENS_CI, self.SENS)  # SENS

    @staticmethod
    def _save_metric_to_file(metric_filename, metric, columns=None, index=None):
        """
        Write single metric data to a CSV file (tab separated).
        Before writing data to files, data are converted to a `pandas.DataFrame`.
        
        Parameters
        ----------
        metric_filename: str
            The name of the output file where to save metrics data
            
        metric: array-like, shape = (n_iterations, n_feature_steps) [typically]
            The 2D data for input metric collected during the whole DAP process.
            
        columns: list
            List of labels for columns of the data frame (1st line in the output file).
            
        index: list
            List of labels to be used as Index in the DataFrame.
        """

        df = pd.DataFrame(metric, columns=columns, index=index)
        df.to_csv(metric_filename, sep='\t')

    def _save_all_metrics_to_file(self, base_output_folder_path, feature_steps, feature_names, sample_names):
        """
        Save all basic metrics to corresponding files.
        
        Parameters
        ----------
        base_output_folder_path: str
            Path to the output folder where files will be saved.
             
        feature_steps: list
            List with all feature steps.
            
        feature_names: list
            List with all the names of the features
            
        sample_names: list
            List of all sample names
        """

        blacklist = [self.RANKINGS, self.PREDS]
        for key in self.BASE_METRICS:
            if key not in blacklist:
                metric_values = self.metrics[key]
                self._save_metric_to_file(os.path.join(base_output_folder_path, 'metric_{}.txt'.format(key)),
                                          metric_values, feature_steps)

        # Save Ranking
        self._save_metric_to_file(os.path.join(base_output_folder_path, 'metric_{}.txt'.format(self.RANKINGS)),
                                  self.metrics[self.RANKINGS], feature_names)

        # Save Metrics for Predictions
        for i_step, step in enumerate(feature_steps):
            for key in blacklist[1:]:  # exclude RANKING, already saved
                self._save_metric_to_file(os.path.join(base_output_folder_path,
                                                       'metric_{}_fs{}.txt'.format(key, step)),
                                          self.metrics[key][:, i_step, :], sample_names)

        # Save Confidence Intervals Metrics
        # NOTE: All metrics will be saved together into a unique file.
        ci_metric_values = list()
        metric_names = list()  # collect names to become indices of resulting pd.DataFrame
        for metric_key in self.CI_METRICS:
            metric_values = self.metrics[metric_key]
            nb_features_steps = metric_values.shape[0]
            metric_names.extend(['{}-{}'.format(metric_key, s) for s in range(nb_features_steps)])
            ci_metric_values.append(metric_values)
        ci_metric_values = np.vstack(ci_metric_values)
        self._save_metric_to_file(os.path.join(base_output_folder_path, 'CI_All_metrics.txt'),
                                  ci_metric_values, columns=['Mean', 'Lower', 'Upper'],
                                  index=metric_names)

    def _save_test_metrics_to_file(self, base_output_folder_path):
        """
        Save all basic metrics to corresponding files.

        Parameters
        ----------
        base_output_folder_path: str
            Path to the output folder where files will be saved.

        """

        column_name = []
        self.metrics[self.TEST].pop(self.PREDS)
        test_metrics = np.zeros((1, len(self.metrics[self.TEST].keys())))
        for i, item in enumerate(self.metrics[self.TEST].items()):
            key, value = item
            test_metrics[0, i] = value
            column_name.append(key)

        self._save_metric_to_file(os.path.join(base_output_folder_path, 'metric_test.txt'),
                                  test_metrics, column_name)

    @staticmethod
    def _generate_feature_steps(nb_features):
        """
        Generate the feature_steps, i.e. the total number
        of features to select at each step in the Cross validation.
        
        Total features for each step are collected according to the 
        total number of features, and feature percentages specified in settings
        (see: settings.feature_ranges).
        
        Parameters
        ----------
        nb_features: int
            The total number of feature
            
        Returns
        -------
        k_features_indices: list
            The number of features (i.e. indices) to consider 
            in slicing features at each (feature) step.
        """
        k_features_indices = list()
        if settings.use_top_feature:
            k_features_indices.append(1)
        for percentage in settings.feature_ranges:
            k = np.ceil((nb_features * percentage) / 100).astype(np.int)
            k_features_indices.append(k)

        return k_features_indices

    # ====== DAP Ops Methods ======
    #
    def _train_validation_split(self, training_indices, validation_indices):
        """DAP standard implementation for train-validation splitting

        Parameters
        ----------
        training_indices: array-like
            Indices of samples in training set

        validation_indices: array-like
            Indices of samples in the validation set
        """
        Xs_tr, Xs_val = self.X[training_indices], self.X[validation_indices]
        ys_tr, ys_val = self.y[training_indices], self.y[validation_indices]
        return (Xs_tr, Xs_val), (ys_tr, ys_val)

    def _get_feature_scaler(self):
        """
        Retrieves the Scaler object corresponding to what 
        specified in settings.
        
        Returns
        -------    
        sklearn.base.TransformerMixin
            Scikit-learn feature scaling estimator, already trained on
            input training data, and so ready to be applied on validation data.
            
        Raises
        ------
        ValueError if a wrong/not supported feature scaling method has been
        specified in settings.
        """
        # This mimics what is done by Keras when accepting
        # string parameters (e.g. `optimizer='los'`).
        try:
            self.feature_scaler = globals().get(self.feature_scaler)
        except:
            raise ValueError('No Feature Scaling Method found!')

    def _apply_scaling(self, X_train, X_validation):
        """
        Apply feature scaling on training and validation data.
        
        Parameters
        ----------
        X_train: array-like, shape = (n_samples, n_features)
            Training data to `fit_transform` by the selected feature scaling method
            
        X_validation: array-like, shape = (n_samples, n_features)
            Validation data to `transform` by the selected feature scaling method

        Returns
        -------
        X_train_scaled: array-like, shape = (n_samples, n_features)
            Training data with features scaled according to the selected
            scaling method/estimator.
            
        X_val_scaled: array-like, shape = (n_samples, n_features)
            Validation data with features scaled according to the selected
            scaling method/estimator.
        
        """
        if isinstance(self.feature_scaler, str):
            scaler = self._get_feature_scaler()
        else:
            scaler = self.feature_scaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_validation)
        return X_train_scaled, X_val_scaled

    def _apply_feature_ranking(self, X_train, y_train, seed=np.random.seed(1234)):
        """
        Compute the ranking of the features as required. It raises an exception 
        if a wrong rank_method name is specified in settings. 
         
        Parameters
        ----------
        X_train: array-like, shape = (n_samples, n_features)
            Training data matrix
            
        y_train: array-like, shape = (n_samples)
            Training labels
            
        seed: int (default: np.random.rand(1234))
            Integer seed to use in random number generators.

        Returns
        -------
        ranking: array-like, shape = (n_features, )
            features ranked by the selected ranking method
        """
        if isinstance(self.feature_scaler, str):
            # This mimics what is done by Keras when accepting
            # string parameters (e.g. `optimizer='los'`).
            try:
                self.feature_ranker = globals().get(self.feature_ranker)
            except:
                raise ValueError('No Feature Ranking Method found!')

        ranking = self.feature_ranker(X_train, y_train, seed=seed)
        return ranking

    def _select_ranked_features(self, ranked_feature_indices, X_train, X_validation=None):
        """Filter features according to input ranking
        
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

        X_train_fs = X_train[:, ranked_feature_indices]
        if X_validation is not None:
            X_val_fs = X_validation[:, ranked_feature_indices]
            return X_train_fs, X_val_fs
        return X_train_fs

    def _fit_predict(self, model, X_train, y_train, X_validation, y_validation=None):
        """
        Core method to generate metrics on (feature-step) data by fitting 
        the input machine learning model and predicting on validation data.
        on validation data.
        
        Parameters
        ----------
        model: sklearn.base.BaseEstimator
            Scikit-learn Estimator object
            
        X_train: array-like, shape = (n_samples, n_features)
            Training data of the current feature step
            
        y_train: array-like, shape = (n_samples, )
            Training targets
            
        X_validation: array-like, shape = (n_samples, n_features)
            Validation data of the current feature step
            
        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation targets (None by default as it is not used in predict)

        Returns
        -------
        predicted_classes: array-like, shape = (n_samples, )
            Array containing the class predictions generated by the model
            
        predicted_class_probs: array-like, shape = (n_samples, n_classes)
            Array containing the prediction probabilities estimated by the model 
            for each of the considered targets (i.e. classes)
            
        extra_metrics:
            List of extra metrics to log during execution.
        """

        X_train = self._prepare_data(X_train, training_data=True)
        y_train = self._prepare_targets(y_train, training_labels=True)
        model, extra_metrics = self._fit(model, X_train, y_train)

        X_validation = self._prepare_data(X_validation, training_data=False)
        predicted_classes, predicted_class_probs = self._predict(model, X_validation)
        return predicted_classes, predicted_class_probs, extra_metrics

    def _prepare_data(self, X, training_data=True):
        """
        Preparation of data before training/inference.
        Current implementation (default behaviour) does not apply
        any operation (i.e. Input data remains unchanged!)
        
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
        array-like, shape = (n_samples, n_features)
            Input data unchanged (Identity)
        """
        return X

    def _prepare_targets(self, y, training_labels=True):
        """
        Preparation of targets before training/inference.
        Current implementation only checks whether categorical one-hot encoding 
        is required on labels. Otherwise, input labels remain unchanged. 
        
        Parameters
        ----------
        y: array-like, shape = (n_samples, )
            array of targets for each sample.
            
        training_labels: bool (default: True)
            Flag indicating whether input targets refers to training data or not.
            This flag is included as it may be required to prepare labels
            differently depending they refers to training or validation data.

        Returns
        -------
        y : array-like, shape = (n_samples, )
            Array of targets whose dimensions will be unchanged, if no encoding has been applied),
            or (samples x nb_classes) otherwise.

        """
        if self.to_categorical:
            y = np_utils.to_categorical(y, self.experiment_data.nb_classes)
        return y

    def _fit(self, model, X_train, y_train, X_validation=None, y_validation=None):
        """
        Default implementation of the training (`fit`) step of input model 
        considering scikit-learn Estimator API.
        
        Parameters
        ----------
        model: sklearn.base.BaseEstimator
            Generic Scikit-learn Classifier
            
        X_train: array-like, shape = (n_samples, n_features)
            Training data
            
        y_train: array-like, shape = (n_samples, )
            Training labels
            
        X_validation: array-like, shape = (n_samples, n_features) - default: None
            Validation data to be used in combination with Training data.
            This parameter has been included to maintain compatibility with
            keras.models.Model.fit API allowing to pass validation data in `fit`.
            
        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation labels to be used in combination with validation data.
            This parameter has been included to maintain compatibility with
            keras.models.Model.fit API allowing to pass validation data in `fit`.

        Returns
        -------
        model:    the trained model
        
        extra_metrics: dict
            List of extra metrics to be logged during execution. Default: None
        """
        model = model.fit(X_train, y_train)
        extra_metrics = None  # No extra metrics is returned by default
        return model, extra_metrics

    def _predict(self, model, X_validation, y_validation=None, **kwargs):
        """
        Default implementation of the inference (`predict`) step of input model 
        considering scikit-learn Estimator API.
        
        Parameters
        ----------
        model: sklearn.base.BaseEstimator
            Classifier sklearn model implementing Estimator API
            
        X_validation: array-like, shape = (n_samples, n_features)
            Validation data
            
        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation labels. None by default as it is not used.
            
        Other Parameters
        ----------------   
        
        kwargs: dict
            Additional arguments to pass to the inference

        Returns
        -------
        predicted_classes: array-like, shape = (n_samples, )
            Array containing the class predictions generated by the model
            
        predicted_class_probs: array-like, shape = (n_samples, n_classes)
            Array containing the prediction probabilities estimated by the model 
            for each of the considered targets (i.e. classes)
        """
        predicted_classes = model.predict(X_validation)
        if hasattr(model, 'predict_proba'):
            predicted_class_probs = model.predict_proba(X_validation)
        else:
            predicted_class_probs = None
        return predicted_classes, predicted_class_probs

    def _train_best_model(self, k_feature_indices, seed=None):
        """
        Train a new model on the best set of features resulting **after** the entire
        Cross validation process has been completed.
        
        Parameters
        ----------
        k_feature_indices: list
            List containing the total number of features to consider at each 
            feature-step.
            
        seed: int
            Random seed to be used for new feature ranking.

        Returns
        -------
        dap_model: sklearn.base.BaseEstimator
            The estimator object fit on the whole training set (i.e. (self.X, self.y) )
            
        extra_metrics: dict
            Dictionary containing information about extra metrics to be monitored during the 
            process and so to be saved, afterwards.
        """

        # Get Best Feature Step (i.e. no. of features to use)
        mcc_means = self.metrics[self.MCC_CI][:, 0]
        max_index = np.argmax(mcc_means)
        best_nb_features = k_feature_indices[max_index]

        # update contextual information
        self._nb_features = best_nb_features  # set this attr. to possibly reference the model
        self._iteration_step_nb = self.cv_n + 1  # last step
        self._feature_step_nb = self.cv_n + 1  # flag value indicating last step

        # Set Training data
        X_train = self.X
        y_train = self.y

        # 2.1 Apply Feature Scaling (if needed)
        if self.apply_feature_scaling:
            X_train = self.feature_scaler.fit_transform(X_train)
            # Note: self.feature_scaler already stores the reference to scaler object

        # 3. Apply Feature Ranking
        if settings.use_borda:
            ranking, _, _ = mlpy.borda_count(self.metrics[self.RANKINGS])
        else:
            ranking = self._apply_feature_ranking(X_train, y_train, seed=seed)

        self._best_feature_ranking = ranking
        Xs_train_fs = self._select_ranked_features(ranking[:best_nb_features], X_train)

        # 4. Fit the model
        model = self.ml_model
        # 4.1 Prepare data
        Xs_train_fs = self._prepare_data(Xs_train_fs)
        y_train = self._prepare_targets(y_train)

        model, extra_metrics = self._fit(model, Xs_train_fs, y_train)

        return model, extra_metrics

    # ===========================================================

    def run(self, verbose=False):
        """
        Implement the entire Data Analysis Plan Main loop.
        
        Parameters
        ----------
        verbose: bool
            Flag specifying verbosity level (default: False, i.e. no output produced)

        Returns
        -------
        dap_model
            The estimator object fit on the whole training set (i.e. (self.X, self.y) )
            Note: The type or returned `dap_model` may change depending on
            different DAP subclasses (e.g. A `keras.models.Model` is returned by the 
            `DeepLearningDap` subclass).
            
        """
        base_output_folder = self._get_output_folder()

        # Select K-features according to the resulting ranking
        k_features_indices = self._generate_feature_steps(self.experiment_data.nb_features)

        for runstep in range(self.cv_n):

            # Save contextual information
            self._runstep_nb = runstep

            # 1. Generate K-Folds
            if self.is_stratified:
                kfold_indices = mlpy.cv_kfold(n=self.experiment_data.nb_samples,
                                              k=self.cv_k, strat=self.y, seed=runstep)
            else:
                kfold_indices = mlpy.cv_kfold(n=self.experiment_data.nb_samples,
                                              k=self.cv_k, seed=runstep)

            if verbose:
                print('=' * 80)
                print('{} over {} experiments'.format(runstep + 1, self.cv_n))
                print('=' * 80)

            for fold, (training_indices, validation_indices) in enumerate(kfold_indices):
                # Save contextual information
                self._iteration_step_nb = runstep * self.cv_k + fold
                self._fold_nb = fold
                self._fold_training_indices = training_indices
                self._fold_validation_indices = validation_indices

                if verbose:
                    print('=' * 80)
                    print('Experiment: {} - Fold {} over {} folds'.format(runstep + 1, fold + 1, self.cv_k))

                # 2. Split data in Training and Validation sets
                (X_train, X_validation), (y_train, y_validation) = self._train_validation_split(training_indices,
                                                                                                validation_indices)

                # 2.1 Apply Feature Scaling (if needed)
                if self.apply_feature_scaling:
                    if verbose:
                        print('-- centering and normalization using: {}'.format(self.feature_scaling_name))
                    X_train, X_validation = self._apply_scaling(X_train, X_validation)

                # 3. Apply Feature Ranking
                if verbose:
                    print('-- ranking the features using: {}'.format(self.feature_ranking_name))
                    print('=' * 80)
                ranking = self._apply_feature_ranking(X_train, y_train, seed=runstep)
                self.metrics[self.RANKINGS][self._iteration_step_nb] = ranking  # store ranking

                # 4. Iterate over Feature Steps
                for step, nb_features in enumerate(k_features_indices):
                    # 4.1 Select Ranked Features
                    X_train_fs, X_val_fs = self._select_ranked_features(ranking[:nb_features], X_train, X_validation)

                    # Store contextual info about current number of features used in this iteration and
                    # corresponding feature step.

                    # Note: The former will be effectively used in the `DeepLearningDAP` subclass to
                    # properly instantiate the Keras `InputLayer`.
                    self._nb_features = nb_features
                    self._feature_step_nb = step

                    # 5. Fit and Predict\
                    model = self.ml_model
                    # 5.1 Train the model and generate predictions (inference)
                    predicted_classes, predicted_class_probs, extra_metrics = self._fit_predict(model,
                                                                                                X_train_fs, y_train,
                                                                                                X_val_fs, y_validation)
                    self._compute_step_metrics(validation_indices, y_validation,
                                               predicted_classes, predicted_class_probs, **extra_metrics)

                    if verbose:
                        print("MCC: {}".format(
                            self.metrics[self.MCC][self._iteration_step_nb, self._feature_step_nb]))

        # Compute Confidence Intervals for all target metrics
        self._compute_metrics_confidence_intervals(k_features_indices)
        # Save All Metrics to File
        self._save_all_metrics_to_file(base_output_folder, k_features_indices,
                                       self.experiment_data.feature_names, self.experiment_data.sample_names)

        if verbose:
            print('=' * 80)
            print('Fitting and predict best model')
            print('=' * 80)

        dap_model, extra_metrics = self._train_best_model(k_features_indices, seed=self.cv_n + 1)
        if extra_metrics:
            self._compute_extra_step_metrics(extra_metrics)

        # Finally return the trained model
        return dap_model

    def predict_on_test(self, model, X_test, Y_test):
        """"""

        feature_ranking = self._best_feature_ranking[:self._nb_features]

        X_test = self._select_ranked_features(feature_ranking, X_test)
        # Prepare data before predict
        X_test = self._prepare_data(X_test)
        predicted_classes, predicted_class_probs = self._predict(model, X_test)
        self._compute_test_metrics(Y_test, predicted_classes, predicted_class_probs)

        output_dir = self._get_output_folder()
        self._save_test_metrics_to_file(output_dir)


class DeepLearningDAP(DAP):
    """DAP Specialisation for plans using Deep Neural Network Models as Learning models"""

    # Neural Network Specific Metric Keys
    NN_VAL_ACC = 'NN_val_acc'
    NN_ACC = 'NN_acc'
    NN_VAL_LOSS = 'NN_val_loss'
    NN_LOSS = 'NN_loss'
    HISTORY = 'model_history'
    NETWORK_METRICS = [NN_VAL_ACC, NN_ACC, NN_VAL_LOSS, NN_LOSS]

    def __init__(self, experiment):
        super(DeepLearningDAP, self).__init__(experiment=experiment)

        # Set additional attributes from Deep Learning Specific Settings set.
        self.learning_epochs = settings.epochs
        self.batch_size = settings.batch_size
        self.fit_verbose = settings.fit_verbose
        self.fit_callbacks = settings.callbacks

        # extra fit parameters
        self.extra_fit_params = {
            'validation_split': settings.validation_split,
            'shuffle': settings.shuffle,
        }
        if settings.initial_epoch:
            self.extra_fit_params['initial_epoch'] = settings.initial_epoch
        if settings.sample_weight:
            self.extra_fit_params['sample_weight'] = settings.sample_weight
        if settings.class_weight:
            self.extra_fit_params['class_weight'] = settings.class_weight

        # Compilation Settings
        self.optimizer = settings.optimizer
        self.loss_function = settings.loss
        self.learning_metrics = settings.metrics
        self.loss_weights = settings.loss_weights
        self.extra_compile_params = settings.extra_compilation_parameters

        # Model Cache - one model reference per feature step
        self._model_cache = {}
        self._do_serialisation = True  # Checks whether model serialisation works

    @property
    def ml_model(self):
        """Keras Model to be used in DAP.

        Note: Differently from "standard" DAP, bound to sklearn estimators, 
        a **brand new** model (network) must be returned at each call, 
        namely with a brand new set of weights each time this property is called.
        """
        # if not self.ml_model_:
        #     self.ml_model_ = self._create_ml_model()
        # else:
        #     pass  # Random Shuffling of Weights

        # Note: the value of self._nb_features attribute is updated during the main DAP loop,
        # during each iteration, before this !
        cache_key = self._nb_features
        if cache_key in self._model_cache:
            if self._do_serialisation:
                try:
                    from_json = self._model_cache[cache_key]
                    model = model_from_json(from_json, custom_objects=self.custom_layers_objects())
                except Exception:
                    self._do_serialisation = False
                    self._model_cache = dict()  #reset cache
                    model = self._create_ml_model()
            else:
                model = self._create_ml_model()
        else:
            model = self._create_ml_model()
            if self._do_serialisation:
                try:
                    self._model_cache[cache_key] = model.to_json()
                except Exception:
                    # Something went wrong during serialisation
                    self._do_serialisation = False

        self.ml_model_ = model
        return self.ml_model_

    def _create_ml_model(self):
        """Instantiate a new Keras Deep Network to be used in the fit-predict step.
        
        Returns
        -------
        model: keras.models.Model
            The new deep learning Keras model.
        """

        model = self._build_network()

        # Set Compilation Params
        extra_compile_params = {}
        if self.loss_weights:
            extra_compile_params['loss_weights'] = self.loss_weights
        if self.extra_compile_params:
            extra_compile_params.update(**self.extra_compile_params)

        model.compile(loss=self.loss_function, optimizer=self.optimizer,
                      metrics=self.learning_metrics, **extra_compile_params)
        return model

    @staticmethod
    def custom_layers_objects():
        """Utility method to specify references to custom layer objects, 
        to be used in de-serialising models.
        
        Returns
        -------
        dic
            dictionary mapping names (strings) to custom classes or functions to be
            considered during deserialization. 
            None by default (no custom layer)
        """
        return None

    @abstractmethod
    def _build_network(self):
        """Abstract method that must be implemented by subclasses to actually
        build the Neural Network graph of layers. This method must return a new
        keras.models.Model object."""

    # ==== Overriding of Utility Methods ====
    #
    def _prepare_metrics_array(self):
        """
        Specialise metrics with extra DNN specific metrics.
        """
        metrics = super(DeepLearningDAP, self)._prepare_metrics_array()

        metrics_shape = (self.iteration_steps, self.feature_steps)
        metrics[self.NN_LOSS] = np.zeros(metrics_shape + (settings.epochs,), dtype=np.float)
        metrics[self.NN_VAL_LOSS] = np.zeros(metrics_shape + (settings.epochs,), dtype=np.float)
        metrics[self.NN_ACC] = np.zeros(metrics_shape + (settings.epochs,), dtype=np.float)
        metrics[self.NN_VAL_ACC] = np.zeros(metrics_shape + (settings.epochs,), dtype=np.float)
        return metrics

    def _compute_extra_step_metrics(self, validation_indices=None, validation_labels=None,
                                    predicted_labels=None, predicted_class_probs=None, **extra_metrics):
        """
        Compute extra additional step metrics, specific to Neural Network leaning resulting from 
        Keras Models.
        In details, kwargs is expected to contain a key for 'model_history'.
        
        Parameters
        ----------
        validation_indices: array-like, shape = (n_samples, )
            Indices of validation samples
            
        validation_labels: array-like, shape = (n_samples, )
            Array of labels for samples in the validation set
            
        predicted_labels: array-like, shape = (n_samples, )
            Array of predicted labels for each sample in the validation set
            
        predicted_class_probs: array-like, shape = (n_samples, n_classes)
            matrix containing the probability for each class and for each sample in the
            validation set
            
        extra_metrics: dict
            By default, the list of extra metrics will contain model history resulting after training.
            See Also: `_fit_predict` method.
        """

        # Compute Extra Metrics
        model_history = extra_metrics.get(self.HISTORY, None)
        if model_history:
            standard_metrics = ['loss', 'val_loss', 'acc', 'val_acc']
            metric_keys = [self.NN_LOSS, self.NN_VAL_LOSS, self.NN_ACC, self.NN_VAL_ACC]
            for history_key, metric_name in zip(standard_metrics, metric_keys):
                metric_values = model_history.history.get(history_key, None)
                if metric_values:
                    if len(metric_values) < settings.epochs:  # early stopping case
                        values = np.zeros(shape=settings.epochs)
                        values[:len(metric_values)] = metric_values
                    else:
                        values = np.array(metric_values)
                    self.metrics[metric_name][self._iteration_step_nb, self._feature_step_nb] = values

    def _compute_extra_test_metrics(self, test_labels=None, predicted_labels=None,
                                    predicted_class_probs=None, **extra_metrics):
        """
        Compute extra additional step metrics, specific to Neural Network leaning resulting from 
        Keras Models.
        In details, kwargs is expected to contain a key for 'model_history'.

        Parameters
        ----------

        test_labels: array-like, shape = (n_samples, )
            Array of labels for samples in the validation set

        predicted_labels: array-like, shape = (n_samples, )
            Array of predicted labels for each sample in the validation set

        predicted_class_probs: array-like, shape = (n_samples, n_classes)
            matrix containing the probability for each class and for each sample in the
            validation set

        extra_metrics: dict
            By default, the list of extra metrics will contain model history resulting after training.
            See Also: `_fit_predict` method.
        """

        # Compute Extra Metrics
        model_history = extra_metrics.get(self.HISTORY, None)
        if model_history:
            standard_metrics = ['loss', 'val_loss', 'acc', 'val_acc']
            metric_keys = [self.NN_LOSS, self.NN_VAL_LOSS, self.NN_ACC, self.NN_VAL_ACC]
            for history_key, metric_name in zip(standard_metrics, metric_keys):
                metric_values = model_history.history.get(history_key, None)
                if metric_values:
                    if len(metric_values) < settings.epochs:  # early stopping case
                        values = np.zeros(shape=settings.epochs)
                        values[:len(metric_values)] = metric_values
                    else:
                        values = np.array(metric_values)
                    self.metrics[self.TEST][metric_name] = values

    def _save_all_metrics_to_file(self, base_output_folder_path, feature_steps, feature_names, sample_names):
        """
        Specialised implementation for Deep learning models, saving to files also 
        Network history losses and accuracy.
        
        Parameters
        ----------
        base_output_folder_path: str
            Path to the output folder where files will be saved.
             
        feature_steps: list
            List with all feature steps.
            
        feature_names: list
            List with all the names of the features
            
        sample_names: list
            List of all sample names
        """

        # Save Base Metrics and CI Metrics as in the Classical DAP
        super(DeepLearningDAP, self)._save_all_metrics_to_file(base_output_folder_path, feature_steps,
                                                               feature_names, sample_names)

        # Save Deep Learning Specific Metrics
        epochs_names = ['epoch {}'.format(e) for e in range(self.learning_epochs)]
        for i_step, step in enumerate(feature_steps):
            for metric_key in self.NETWORK_METRICS:
                self._save_metric_to_file(os.path.join(base_output_folder_path,
                                                       'metric_{}_fs{}.txt'.format(step, metric_key)),
                                          self.metrics[metric_key][:, i_step, :], epochs_names)

    # ==== Overriding of DAP Ops Methods ====
    #
    def _fit_predict(self, model, X_train, y_train, X_validation, y_validation=None):
        """
        Core method to generate metrics on (feature-step) data by fitting 
        the input deep learning model and predicting on validation data.
        on validation data.
        
        Note: The main difference in the implementation of this method relies on the 
        fact that the `_fit` method is provided also with validation data (and targets)
        to be fed into the actual `model.fit` method.

        Parameters
        ----------
        model: keras.models.Model
            Deep Learning network model

        X_train: array-like, shape = (n_samples, n_features)
            Training data of the current feature step

        y_train: array-like, shape = (n_samples, )
            Training targets

        X_validation: array-like, shape = (n_samples, n_features)
            Validation data of the current feature step

        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation targets (None by default as it is not used in predict)

        Returns
        -------
        predicted_classes: array-like, shape = (n_samples, )
            Array containing the class predictions generated by the model

        predicted_class_probs: array-like, shape = (n_samples, n_classes)
            Array containing the prediction probabilities estimated by the model 
            for each of the considered targets (i.e. classes)

        extra_metrics: dict
            List of extra metrics to log during execution. By default, the 
            network history will be returned.
            
        See Also
        --------
        DeepLearningDAP._fit(...)
        """

        # Prepare Data
        X_train = self._prepare_data(X_train, training_data=True)
        y_train = self._prepare_targets(y_train, training_labels=True)
        X_validation = self._prepare_data(X_validation, training_data=False)
        y_validation = self._prepare_targets(y_validation, training_labels=False)

        model, extra_metrics = self._fit(model, X_train, y_train,
                                         X_validation=X_validation, y_validation=y_validation)
        predicted_classes, predicted_class_probs = self._predict(model, X_validation)
        return predicted_classes, predicted_class_probs, extra_metrics

    def _fit(self, model, X_train, y_train, X_validation=None, y_validation=None):
        """
        Default implementation of the training (`fit`) step for an input Keras 
        Deep Learning Network model.

        Parameters
        ----------
        model: keras.models.Model
            Deep Learning network model

        X_train: array-like, shape = (n_samples, n_features)
            Training data

        y_train: array-like, shape = (n_samples, )
            Training labels
        
        X_validation: array-like, shape = (n_samples, n_features) - default: None
            Validation data to be used in combination with Training data.
            This parameter has been included to maintain compatibility with
            keras.models.Model.fit API allowing to pass validation data in `fit`.
            
        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation labels to be used in combination with validation data.
            This parameter has been included to maintain compatibility with
            keras.models.Model.fit API allowing to pass validation data in `fit`.

        Returns
        -------
        model: the model along with learned weights resulting from the actual training
            process.

        extra_metrics: dict
            List of extra metrics to be logged during execution. Default: `model_history`
        """

        # Setup extra fit parameters from settings
        if X_validation is not None and y_validation is not None:
            self.extra_fit_params['validation_data'] = (X_validation, y_validation)
        else:
            self.extra_fit_params.pop('validation_data', None)

        model_filename = '{}_{}_model.hdf5'.format(self._iteration_step_nb, self._nb_features)
        base_output_folder = self._get_output_folder()
        model_filename = os.path.join(base_output_folder, model_filename)
        callbacks = [ModelCheckpoint(filepath=model_filename, save_best_only=True, save_weights_only=True), ]
        if self.fit_callbacks:
            callbacks.extend(self.fit_callbacks)
            if X_validation is None:
                for callback in callbacks:
                    if hasattr(callback, 'monitor') and callback.monitor == 'val_loss':
                        callback.monitor = 'loss'

        if self.fit_verbose != 0:
            print('Experiment {} - fold {}'.format(self._runstep_nb + 1, self._fold_nb + 1))
            print('Step {}, working with {} features'.format(self._feature_step_nb + 1, self._nb_features))

        model_history = model.fit(X_train, y_train, epochs=self.learning_epochs,
                                  batch_size=self.batch_size, verbose=self.fit_verbose,
                                  callbacks=callbacks, **self.extra_fit_params)
        extra_metrics = {
            self.HISTORY: model_history
        }
        return model, extra_metrics

    def _predict(self, model, X_validation, y_validation=None, **kwargs):
        """
        Default implementation of the inference (`predict`) step of input 
        Keras model.

        Parameters
        ----------
        model: keras.models.Model
            Deep Learning network model

        X_validation: array-like, shape = (n_samples, n_features)
            Validation data

        y_validation: array-like, shape = (n_samples, ) - default: None
            Validation labels. None by default as it is not used.
        
        Other Parameters
        ----------------
        
        kwargs: dict
            Additional arguments to pass to the inference

        Returns
        -------
        predicted_classes: array-like, shape = (n_samples, )
            Array containing the class predictions generated by the model

        predicted_class_probs: array-like, shape = (n_samples, n_classes)
            Array containing the prediction probabilities estimated by the model 
            for each of the considered targets (i.e. classes)
        """

        predicted_class_probs = model.predict(X_validation)
        predicted_classes = predicted_class_probs.argmax(axis=-1)
        return predicted_classes, predicted_class_probs
