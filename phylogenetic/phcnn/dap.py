import csv
import os

import numpy as np
import mlpy

from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Input
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense, Flatten, Dropout
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


def _prepare_metrics_array(cv_k, cv_n, nb_features, nb_samples):

    iterations = cv_k * cv_n
    feature_sets = len(settings.feature_selection_percentage)
    if settings.include_top_feature:
        feature_sets += 1
    metrics_shape = (iterations, feature_sets)
    metrics = {
        'ranking': np.empty((iterations, nb_features), dtype=np.int),
        'NPV': np.empty(metrics_shape),
        'PPV': np.empty(metrics_shape),
        'SENS': np.empty(metrics_shape),
        'SPEC': np.empty(metrics_shape),
        'MCC': np.empty(metrics_shape),
        'AUC': np.empty(metrics_shape),
        'DOR': np.empty(metrics_shape),
        'ACC': np.empty(metrics_shape),
        'ACCint': np.empty(metrics_shape),
        'PREDS': np.empty(metrics_shape + (nb_samples,), dtype=np.int),
        'REALPREDS_0': np.empty(metrics_shape + (nb_samples,), dtype=np.float),
        'REALPREDS_1': np.empty(metrics_shape + (nb_samples,), dtype=np.float)
    }

    # Initialize Flag Values
    metrics['PREDS'][:,:,:] = -10
    metrics['REALPREDS_0'][:, :, :] = -10
    metrics['REALPREDS_1'][:, :, :] = -10

    return metrics


class _OptimizerNotFound(Exception):
    pass


def _get_optimizer(selected_optimizer):
    """
    Configure a optimizer. It raises an exception if a wrong optimizer name is required.

    :param selected_optimizer: string containing the name of the optimizer that we want to configure 
    :return: The optimizer and a dictionary containing the name and the parameter of the optimizer 
   """

    if selected_optimizer == 'adam':
        epsilon = 1e-08
        lr = 0.001
        beta_1 = 0.9
        beta_2 = 0.999
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
        optimizer_configuration = {'name': 'adam',
                                   'lr': lr,
                                   'beta_1': beta_1,
                                   'beta_2': beta_2,
                                   'epsilon': epsilon
                                   }
    elif selected_optimizer == 'rmsprop':
        epsilon = 1e-08
        lr = 0.001
        rho = 0.9
        optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon)
        optimizer_configuration = {'name': 'rmsprop',
                                   'lr': lr,
                                   'rho': rho,
                                   'epsilon': epsilon}
    elif selected_optimizer == 'sgd':
        nesterov = True
        lr = 0.001
        momentum = 0.9
        decay = 1e-06
        optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        optimizer_configuration = {'name': 'sgd',
                                   'lr': lr,
                                   'momentum': momentum,
                                   'decay': decay,
                                   'nesterov': nesterov}
    else:
        raise _OptimizerNotFound("The only supported optimizer are adam, rmsprop, sgd")

    return optimizer, optimizer_configuration


class _ScalingNotFound(Exception):
    pass


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
    if selected_scaling == 'norm_l2':
        xs_train_scaled, m_train, r_train = norm_l2(xs_tr)
        xs_test_scaled, _, _ = norm_l2(xs_ts, m_train, r_train)
    elif selected_scaling == 'std':
        scaler = preprocessing.StandardScaler(copy=False)
        xs_train_scaled = scaler.fit_transform(xs_tr)
        xs_test_scaled = scaler.transform(xs_ts)
    elif selected_scaling == 'minmax':
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        xs_train_scaled = scaler.fit_transform(xs_tr)
        xs_test_scaled = scaler.transform(xs_ts)
    elif selected_scaling == 'minmax0':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
        xs_train_scaled = scaler.fit_transform(xs_tr)
        xs_test_scaled = scaler.transform(xs_ts)
    else:
        raise _ScalingNotFound("The only supported scaling are norm_l2, std, minmax, minmax0")

    return xs_train_scaled, xs_test_scaled


class _RankingNotFound(Exception):
    pass


def _get_ranking(xs_tr, ys_tr, rank_method='ReliefF', seed=None):
    """
    Compute the ranking of the features as required. It raises an exception 
    if a wrong rank_method name is required.  
    
    :param xs_tr: Training data of shape (number of training samples, number of features)
    :param ys_tr: Training labels
    :param rank_method: a string representing the selected rank method
    :return: 
    """
    if rank_method == 'random':
        ranking = np.arange(xs_tr.shape[1])
        np.random.seed(seed)
        np.random.shuffle(ranking)
    elif rank_method == 'ReliefF':
        relief = ReliefF(settings.relief_k, seed=seed)
        relief.learn(xs_tr, ys_tr)
        w = relief.w()
        ranking = np.argsort(w)[::-1]
    elif rank_method == 'KBest':
        selector = SelectKBest(k=settings.kbest)
        selector.fit(xs_tr, ys_tr)
        ranking = np.argsort(-np.log10(selector.pvalues_))[::-1]
    else:
        raise _RankingNotFound("The only supported Ranking are random, ReliefF, KBest")

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


def _save_metrics_on_file(base_output_name, metrics):

    np.savetxt(base_output_name + "_allmetrics_NPV.txt",
               metrics['NPV'], fmt='%.4f', delimiter='\t')
    np.savetxt(base_output_name + "_allmetrics_PPV.txt",
               metrics['PPV'], fmt='%.4f', delimiter='\t')
    np.savetxt(base_output_name + "_allmetrics_SENS.txt",
               metrics['SENS'], fmt='%.4f', delimiter='\t')
    np.savetxt(base_output_name + "_allmetrics_SPEC.txt",
               metrics['SPEC'], fmt='%.4f', delimiter='\t')
    np.savetxt(base_output_name + "_allmetrics_MCC.txt",
               metrics['MCC'], fmt='%.4f', delimiter='\t')
    np.savetxt(base_output_name + "_allmetrics_AUC.txt",
               metrics['AUC'], fmt='%.4f', delimiter='\t')
    np.savetxt(base_output_name + "_allmetrics_DOR.txt",
               metrics['DOR'], fmt='%.4f', delimiter='\t')
    np.savetxt(base_output_name + "_allmetrics_ACC.txt",
               metrics['ACC'], fmt='%.4f', delimiter='\t')


def phylo_cnn(X_train, X_val, C_train, C_val, Y_train, Y_val,
              nb_features, nb_coordinates, nb_classes):
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

    x = Input(shape=(nb_features,), name="data", dtype='float64')
    coordinates = Input(shape=(nb_coordinates, nb_features), name="coordinates", dtype='float64')
    coord = coordinates[0]

    phylo_ngb = PhyloNeighbours(coordinates=coord,
                                nb_neighbors=nb_neighbors,
                                nb_features=coord.shape[1])

    phylo_conv = PhyloConv2D(nb_neighbors=nb_neighbors,
                             filters=nb_filters)
    conv1 = phylo_conv(phylo_ngb(x))
    conv_crd1 = phylo_conv(phylo_ngb(coord))

    conv2, conv_crd2 = _conv_block(conv1, conv_crd1, nb_neighbors=nb_neighbors, filters=nb_filters)
    conv3, _ = _conv_block(conv2, conv_crd2, nb_neighbors=nb_neighbors, filters=nb_filters)

    max = MaxPool2D(pool_size=(1, 2), padding="valid")(conv3)
    flatt = Flatten()(max)
    drop = Dropout(0, 1)(Dense(units=64)(flatt))
    output = Dense(units=nb_classes, kernel_initializer="he_normal",
                   activation="softmax", name='output')(drop)

    model = Model(inputs=[x, coordinates], outputs=output)

    opt, _ = _get_optimizer(settings.optimizer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model_history = model.fit({'data': X_train, 'coordinates': C_train}, {'output': Y_train},
                              epochs=40, verbose=2, batch_size=settings.batch_size,
                              validation_data=({'data': X_val,
                                                'coordinates': C_val}, {'output': Y_val}))
    return model, model_history


def dap(inputs, model_fn=phylo_cnn):

    basefile_name = settings.DISEASE.lower()
    base_output_name = os.path.join(settings.OUTPUT_DIR, '_'.join([basefile_name,
                                                           settings.ml_model,
                                                           settings.feature_ranking_method,
                                                           settings.feature_scaling_method]))

    metrics = _prepare_metrics_array(settings.Cv_K,
                                     settings.Cv_N,
                                     inputs['nb_features'],
                                     inputs['nb_samples'])

    ## Random Shuffling of Training labels

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
            ys_tr, ys_val = inputs['ys'][idx_tr], inputs['ys'][idx_val]

            Coords_tr, Coords_val = inputs['coordinates'][idx_tr], inputs['coordinates'][idx_val]
            Ys_tr_cat = np_utils.to_categorical(ys_tr, inputs['nb_classes'])
            Ys_val_cat = np_utils.to_categorical(ys_val, inputs['nb_classes'])

            Xs_tr, Xs_val = _apply_scaling(Xs_tr, Xs_val, settings.feature_scaling_method)

            ranking = _get_ranking(Xs_tr, ys_tr, settings.feature_ranking_method, seed=n)
            metrics['ranking'][current] = ranking # store ranking

            # Select K-features according to the resulting ranking
            nb_features = inputs['nb_features']

            k_features_indices = list()
            if settings.include_top_feature:
                k_features_indices.append(1)
            for percentage in settings.feature_selection_percentage:
                k = np.floor((nb_features * percentage)/100).astype(np.int) + 1
                k_features_indices.append(k)

            for step, feature_index in enumerate(k_features_indices):
                # Filter Training data
                Xs_tr_sel = Xs_tr[:, ranking[:feature_index]]
                Coords_tr_sel = Coords_tr[:, :, ranking[:feature_index]]

                # Filter Validation data
                Xs_val_sel = Xs_val[:, ranking[:feature_index]]
                # FIXME: We're still wasting a lot of memory!
                Coords_val_sel = Coords_val[:, :, ranking[:feature_index]]
                nb_features_sel = Xs_tr_sel.shape[1]

                model, hstory = model_fn(Xs_tr_sel, Xs_val_sel, Coords_tr_sel, Coords_val_sel,
                                         Ys_tr_cat, Ys_val_cat, nb_features_sel,
                                         inputs['nb_coordinates'], inputs['nb_classes'])

                score, acc = model.evaluate({'data': Xs_val_sel,
                                             'coordinates': Coords_val_sel},
                                            {'output': Ys_val_cat}, verbose=0)

                pv, p = _predict_classes(model, Xs_val_sel, Coords_val_sel, verbose=0)

                pred_mcc = perf.KCCC_discrete(ys_val, p)

                if not settings.quiet:
                    print('Test accuracy: {}'.format(acc))
                    print('Test MCC: {}'.format(pred_mcc))

                metrics['PREDS'][current, step, idx_val] = p
                metrics['REALPREDS_0'][current, step, idx_val] = pv[:, 0]
                metrics['REALPREDS_1'][current, step, idx_val] = pv[:, 1]

                metrics['NPV'][current, step] = perf.npv(ys_val, p)
                metrics['PPV'][current, step] = perf.ppv(ys_val, p)
                metrics['SENS'][current, step] = perf.sensitivity(ys_val, p)
                metrics['SPEC'][current, step] = perf.specificity(ys_val, p)
                metrics['MCC'][current, step] = pred_mcc
                metrics['AUC'][current, step] = roc_auc_score(ys_val, p)
                metrics['DOR'][current, step] = perf.dor(ys_val, p)
                metrics['ACC'][current, step] = perf.accuracy(ys_val, p)
                metrics['ACCint'][current, step] = acc

    # print("Average MCC: {}".format(np.mean(metrics['MCC'], axis=0)))
    # print("Confidence interval for MCC: {}".format(mlpy.bootstrap_ci(metrics['MCC'])))

    # _save_metrics_on_file(base_output_name, metrics)
    #
    # with open(base_output_name + "_preds.txt", 'w+') as f:
    #     writer = csv.writer(f, delimiter='\t', lineterminator='\n')
    #     writer.writerow(inputs['sample_names'])
    #     for row in metrics['PREDS']:
    #         writer.writerow(row.tolist())

    return metrics



