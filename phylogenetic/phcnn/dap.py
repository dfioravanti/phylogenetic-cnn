import os
import sys

import numpy as np
import pandas as pd
import mlpy

from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.callbacks import ModelCheckpoint
from .layers import PhyloConv2D, PhyloNeighbours, _conv_block

from keras.models import Model

from keras.utils import np_utils

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score

from .relief import ReliefF
from .scaling import norm_l2
from . import settings
from . import performance as perf


# -- Metrics Keys
REALPREDS1 = 'REALPREDS_1'
REALPREDS0 = 'REALPREDS_0'
PREDS = 'PREDS'
ACCINT = 'ACCint'
ACC = 'ACC'
DOR = 'DOR'
AUC = 'AUC'
MCC_CI = 'MCC_CI'
MCC = 'MCC'
SPEC = 'SPEC'
SENS = 'SENS'
PPV = 'PPV'
NPV = 'NPV'
RANKINGS = 'ranking'
NN_VAL_ACC = 'NN_val_acc'
NN_ACC = 'NN_acc'
NN_VAL_LOSS = 'NN_val_loss'
NN_LOSS = 'NN_loss'


def _prepare_metrics_array(cv_k, cv_n, nb_features, nb_samples):

    iterations = cv_k * cv_n
    feature_steps = len(settings.feature_selection_percentage)
    if settings.include_top_feature:
        feature_steps += 1
    metrics_shape = (iterations, feature_steps)
    metrics = {
        RANKINGS: np.empty((iterations, nb_features), dtype=np.int),
        NPV: np.empty(metrics_shape),
        PPV: np.empty(metrics_shape),
        SENS: np.empty(metrics_shape),
        SPEC: np.empty(metrics_shape),
        MCC: np.empty(metrics_shape),
        MCC_CI: np.empty((feature_steps, 2)),
        AUC: np.empty(metrics_shape),
        DOR: np.empty(metrics_shape),
        ACC: np.empty(metrics_shape),
        ACCINT: np.empty(metrics_shape),
        PREDS: np.empty(metrics_shape + (nb_samples,), dtype=np.int),
        REALPREDS0: np.empty(metrics_shape + (nb_samples,), dtype=np.float),
        REALPREDS1: np.empty(metrics_shape + (nb_samples,), dtype=np.float)
    }

    # Initialize to Flag Values
    metrics[PREDS][:, :, :] = -10
    metrics[REALPREDS0][:, :, :] = -10
    metrics[REALPREDS1][:, :, :] = -10

    if settings.ml_model == settings.PHCNN:
        metrics[NN_LOSS] = np.zeros(metrics_shape + (settings.epochs,), dtype=K.floatx())
        metrics[NN_VAL_LOSS] = np.zeros(metrics_shape + (settings.epochs,), dtype=K.floatx())
        metrics[NN_ACC] = np.zeros(metrics_shape + (settings.epochs,), dtype=K.floatx())
        metrics[NN_VAL_ACC] = np.zeros(metrics_shape + (settings.epochs,), dtype=K.floatx())

    return metrics


def _generate_feature_steps(nb_features):
    """

    :param inputs: 
    :return: 
    """
    k_features_indices = list()
    if settings.include_top_feature:
        k_features_indices.append(1)
    for percentage in settings.feature_selection_percentage:
        k = np.floor((nb_features * percentage) / 100).astype(np.int) + 1
        k_features_indices.append(k)

    return k_features_indices


def _get_optimizer(selected_optimizer):
    """
    Configure a optimizer. It raises an exception if a wrong optimizer name is required.

    :param selected_optimizer: string containing the name of the optimizer that we want to configure 
    :return: The optimizer and a dictionary containing the name and the parameter of the optimizer 
   """

    if selected_optimizer == settings.ADAM:
        epsilon = settings.adam_epsilon
        lr = settings.adam_lr
        beta_1 = settings.beta_1
        beta_2 = settings.beta_2
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    elif selected_optimizer == settings.RMSPROP:
        epsilon = settings.rmsprop_epsilon
        lr = settings.rmsprop_lr
        rho = settings.rho
        optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon)

    elif selected_optimizer == settings.SGD:
        nesterov = settings.nesterov
        lr = settings.sgd_lr
        momentum = settings.momentum
        decay = settings.decay
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)

    else:
        raise Exception("The only supported optimizer are {}, {}, {}".format(settings.RMSPROP,
                                                                             settings.SGD, settings.ADAM))

    return optimizer


def _apply_scaling(xs_tr, xs_ts, selected_scaling):
    """
    Apply the scaling to the input data AFTER they have been divided in 
    (training, testing) during a cross validation. It raises an exception 
    if a wrong scaler name is required.
    :param xs_tr: Training data
    :param xs_ts: Test data
    :param selected_scaling: a string representing the select scaling
    :return: (xs_training, xs_ts) scaled according to the scaler. 
    """
    if selected_scaling == settings.NORM_L2:
        xs_train_scaled, m_train, r_train = norm_l2(xs_tr)
        xs_test_scaled, _, _ = norm_l2(xs_ts, m_train, r_train)
    elif selected_scaling == settings.STD:
        scaler = preprocessing.StandardScaler(copy=False)
        xs_train_scaled = scaler.fit_transform(xs_tr)
        xs_test_scaled = scaler.transform(xs_ts)
    elif selected_scaling == settings.MINMAX:
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        xs_train_scaled = scaler.fit_transform(xs_tr)
        xs_test_scaled = scaler.transform(xs_ts)
    elif selected_scaling == settings.MINMAX0:
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
        xs_train_scaled = scaler.fit_transform(xs_tr)
        xs_test_scaled = scaler.transform(xs_ts)
    else:
        raise Exception("The only supported scaling are norm_l2, std, minmax, minmax0")

    return xs_train_scaled, xs_test_scaled


def _get_ranking(xs_tr, ys_tr, rank_method=settings.RELIEFF, seed=None):
    """
    Compute the ranking of the features as required. It raises an exception 
    if a wrong rank_method name is required.  
    
    :param xs_tr: Training data of shape (number of training samples, number of features)
    :param ys_tr: Training labels
    :param rank_method: a string representing the selected rank method
    :return: 
    """
    if rank_method == settings.RANDOM:
        ranking = np.arange(xs_tr.shape[1])
        np.random.seed(seed)
        np.random.shuffle(ranking)
    elif rank_method == settings.RELIEFF:
        relief = ReliefF(settings.relief_k, seed=seed)
        relief.learn(xs_tr, ys_tr)
        w = relief.w()
        ranking = np.argsort(w)[::-1]
    elif rank_method == settings.KBEST:
        selector = SelectKBest(k=settings.kbest)
        selector.fit(xs_tr, ys_tr)
        ranking = np.argsort(-np.log10(selector.pvalues_))[::-1]
    else:
        raise Exception("The only supported Ranking are random, ReliefF, KBest")

    return ranking


def _predict_classes(model, Xs, coordinates, batch_size=32, verbose=1):
    """
    Generate class predictions for the input samples batch by batch.
    # Arguments
        Xs: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns
        A numpy array of class predictions for each calss, along with the 
        index of the class at the maximum probability
    """
    prb = model.predict({'data': Xs, 'coordinates': coordinates},
                        batch_size=batch_size, verbose=verbose)

    if prb.shape[-1] > 1:
        return prb, prb.argmax(axis=-1)
    else:
        return prb, (prb > 0.5).astype(np.int)


def _save_metric_to_file(output_fname, metric, columns):
    """
    
    :param output_fname: 
    :param metric: 
    :param columns: 
    :return: 
    """
    df = pd.DataFrame(metric, columns=columns)
    df.to_csv(output_fname, sep='\t')


def _save_all_metrics_to_file(base_output_fname, metrics,
                              feature_steps, feature_names, sample_names):
    """
    
    :param base_output_fname: 
    :param metrics: 
    :param feature_steps: 
    :return: 
    """

    excluded = [RANKINGS, PREDS, REALPREDS1, REALPREDS0, MCC_CI,
                NN_VAL_ACC, NN_ACC, NN_VAL_LOSS, NN_LOSS]
    for key in metrics:
        if not key in excluded:
            metric_values = metrics[key]
            _save_metric_to_file(os.path.join(base_output_fname, 'metric_{}.txt'.format(key)),
                                 metric_values, feature_steps)

    # Save Ranking
    _save_metric_to_file(os.path.join(base_output_fname, 'metric_{}.txt'.format(RANKINGS)),
                         metrics[RANKINGS], feature_names)

    # Save MCC Confidence Intervals
    _save_metric_to_file(os.path.join(base_output_fname, 'metric_{}.txt'.format(MCC_CI)),
                         metrics[MCC_CI], ['Lower', 'Upper'])

    # Save Composed Predictions
    epochs_names = ['epoch {}'.format(e) for e in range(settings.epochs)]
    headers = {
        PREDS: sample_names,
        REALPREDS1: sample_names,
        REALPREDS0: sample_names,
        NN_LOSS: epochs_names,
        NN_VAL_ACC: epochs_names,
        NN_VAL_LOSS: epochs_names,
        NN_ACC: epochs_names
    }
    for istep, step in enumerate(feature_steps):
        for key in [PREDS, REALPREDS0, REALPREDS1, NN_LOSS, NN_ACC, NN_VAL_LOSS, NN_VAL_ACC]:
            header = headers[key]
            _save_metric_to_file(os.path.join(base_output_fname, 'metric_fs{}_{}.txt'.format(step, key)),
                                metrics[key][:, istep, :], header)


def _adjust_dimensions(X_train, X_val, C_train, C_val):
    """
    
    :param X_train: 
    :param X_val: 
    :param C_train: 
    :param C_val: 
    :return: 
    """

    X_train = np.expand_dims(np.expand_dims(X_train, 1), 3)
    X_val = np.expand_dims(np.expand_dims(X_val, 1), 3)
    C_train = np.expand_dims(np.expand_dims(C_train, 2), 4)
    C_val = np.expand_dims(np.expand_dims(C_val, 2), 4)

    return (X_train, X_val), (C_train, C_val)


def phylo_cnn(nb_features, nb_coordinates, nb_classes):
    """

    :param X_train: 
    :param X_val: 
    :param C_train: 
    :param C_val: 
    :param Y_train: 
    :param Y_val: 
    :param nb_features: 
    :param nb_coordinates: 
    :return: 
    """
    nb_neighbors = settings.nb_phylo_neighbours
    if nb_neighbors > nb_features:
        nb_neighbors = nb_features
    nb_filters = settings.nb_convolutional_filters

    x = Input(shape=(1, nb_features, 1), name="data", dtype='float64')
    coordinates = Input(shape=(nb_coordinates, 1, nb_features, 1), name="coordinates", dtype='float64')
    coord = Lambda(lambda c: c[0])(coordinates)

    phylo_ngb = PhyloNeighbours(coordinates=coord,
                                nb_neighbors=nb_neighbors,
                                nb_features=nb_features)

    phylo_conv = PhyloConv2D(nb_neighbors=nb_neighbors,
                             filters=nb_filters)

    conv1 = phylo_conv(phylo_ngb(x))
    conv_crd1 = phylo_conv(phylo_ngb(coord))

    conv2, conv_crd2 = _conv_block(conv1, conv_crd1, nb_neighbors, nb_features, nb_filters)
    conv3, _ = _conv_block(conv2, conv_crd2, nb_neighbors, nb_features, nb_filters)

    max = MaxPool2D(pool_size=(1, 2), padding="valid")(conv3)
    flatt = Flatten()(max)
    drop = Dropout(0, 1)(Dense(units=64)(flatt))
    output = Dense(units=nb_classes, kernel_initializer="he_normal",
                   activation="softmax", name='output')(drop)

    model = Model(inputs=[x, coordinates], outputs=output)

    opt = _get_optimizer(settings.optimizer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def dap(inputs, model_fn=phylo_cnn):

    basefile_name = settings.DISEASE.lower()
    base_output_folder = os.path.join(settings.OUTPUT_DIR, '_'.join([basefile_name, settings.ml_model.lower(),
                                                                    settings.feature_ranking_method.lower(),
                                                                    settings.feature_scaling_method.lower(),
                                                                    str(settings.Cv_N), str(settings.Cv_K)]))
    os.makedirs(base_output_folder, exist_ok=True)

    metrics = _prepare_metrics_array(settings.Cv_K,
                                     settings.Cv_N,
                                     inputs['nb_features'],
                                     inputs['nb_samples'])

    # Select K-features according to the resulting ranking
    k_features_indices = _generate_feature_steps(inputs['nb_features'])

    # Apply for Random Labels
    ys = inputs['ys']
    if settings.use_random_labels:
        np.random.shuffle(ys)

    for n in range(settings.Cv_N):
        idx = mlpy.cv_kfold(n=inputs['nb_samples'],
                            k=settings.Cv_K,
                            strat=inputs['ys'],
                            seed=n)
        print('=' * 80)
        print('{} over {} experiments'.format(n + 1, settings.Cv_N))

        for i, (idx_tr, idx_val) in enumerate(idx):

            current = n * settings.Cv_K + i

            if not settings.quiet:
                print('=' * 80)
                print('{} over {} folds'.format(i + 1, settings.Cv_K))
                print('-- centering and normalization using: {}'.format(settings.feature_scaling_method))
                print('-- ranking the features using: {}'.format(settings.feature_scaling_method))

            Xs_tr, Xs_val = inputs['xs'][idx_tr], inputs['xs'][idx_val]
            ys_tr, ys_val = ys[idx_tr], ys[idx_val]

            Coords_tr, Coords_val = inputs['coordinates'][:Xs_tr.shape[0]], inputs['coordinates'][:Xs_val.shape[0]]
            Ys_tr_cat = np_utils.to_categorical(ys_tr, inputs['nb_classes'])
            Ys_val_cat = np_utils.to_categorical(ys_val, inputs['nb_classes'])

            Xs_tr, Xs_val = _apply_scaling(Xs_tr, Xs_val, settings.feature_scaling_method)

            ranking = _get_ranking(Xs_tr, ys_tr, settings.feature_ranking_method, seed=n)
            metrics[RANKINGS][current] = ranking # store ranking

            for step, feature_index in enumerate(k_features_indices):
                # Filter Training data
                Xs_tr_sel = Xs_tr[:, ranking[:feature_index]]
                Coords_tr_sel = Coords_tr[:, :, ranking[:feature_index]]

                # Filter Validation data
                Xs_val_sel = Xs_val[:, ranking[:feature_index]]
                # FIXME: We're still wasting a lot of memory!
                Coords_val_sel = Coords_val[:, :, ranking[:feature_index]]
                nb_features_sel = Xs_tr_sel.shape[1]

                (Xs_tr_sel, Xs_val_sel), (Coords_tr_sel, Coords_val_sel) = _adjust_dimensions(Xs_tr_sel, Xs_val_sel,
                                                                                             Coords_tr_sel,
                                                                                             Coords_val_sel)

                model = model_fn(nb_features=nb_features_sel, nb_coordinates=inputs['nb_coordinates'],
                                 nb_classes=inputs['nb_classes'])

                model_fname = '{}_{}_model.hdf5'.format(current, settings.feature_selection_percentage[step])
                model_fname = os.path.join(base_output_folder, model_fname)
                model_history = model.fit({'data': Xs_tr_sel, 'coordinates': Coords_tr_sel}, {'output': Ys_tr_cat},
                                          epochs=settings.epochs, verbose=settings.verbose,
                                          batch_size=settings.batch_size,
                                          validation_data=({'data': Xs_val_sel,
                                                            'coordinates': Coords_val_sel}, {'output': Ys_val_cat}),
                                          callbacks=[ModelCheckpoint(filepath=model_fname, save_best_only=True,
                                                                     save_weights_only=True),])

                score, acc = model.evaluate({'data': Xs_val_sel, 'coordinates': Coords_val_sel},
                                            {'output': Ys_val_cat}, verbose=0)

                pv, p = _predict_classes(model, Xs_val_sel, Coords_val_sel, verbose=0)

                pred_mcc = perf.KCCC_discrete(ys_val, p)

                if not settings.quiet:
                    print('Test accuracy: {}'.format(acc))
                    print('Test MCC: {}'.format(pred_mcc))

                # Overwrite only values corresponding to samples in the current (internal) validation set!
                metrics[PREDS][current, step, idx_val] = p
                metrics[REALPREDS0][current, step, idx_val] = pv[:, 0]
                metrics[REALPREDS1][current, step, idx_val] = pv[:, 1]

                metrics[NPV][current, step] = perf.npv(ys_val, p)
                metrics[PPV][current, step] = perf.ppv(ys_val, p)
                metrics[SENS][current, step] = perf.sensitivity(ys_val, p)
                metrics[SPEC][current, step] = perf.specificity(ys_val, p)
                metrics[MCC][current, step] = pred_mcc
                metrics[AUC][current, step] = roc_auc_score(ys_val, p)
                metrics[DOR][current, step] = perf.dor(ys_val, p)
                metrics[ACC][current, step] = perf.accuracy(ys_val, p)
                metrics[ACCINT][current, step] = acc

                if settings.ml_model == 'phcnn':
                    metrics[NN_LOSS][current, step] = model_history.history['loss']
                    metrics[NN_VAL_LOSS][current, step] = model_history.history['val_loss']
                    metrics[NN_ACC][current, step] = model_history.history['acc']
                    metrics[NN_VAL_ACC][current, step] = model_history.history['val_acc']

    # Compute Confidence Intervals
    for step in range(len(k_features_indices)):
        metrics[MCC_CI][step] = mlpy.bootstrap_ci(metrics['MCC'][:, step])

    if not settings.quiet:
        print("Average MCC for each Feature Steps: {}".format(np.mean(metrics[MCC], axis=0)))
        for step in range(len(k_features_indices)):
            print("Confidence interval for MCC at feature step {}: {}".format(step, metrics[MCC_CI][step]))

    # Save All Metrics to File
    _save_all_metrics_to_file(base_output_folder, metrics, k_features_indices,
                              inputs['feature_names'], inputs['sample_names'])

    return metrics




