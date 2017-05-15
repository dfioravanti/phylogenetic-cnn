import numpy as np
import mlpy
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

from .relief import ReliefF
from .scaling import norm_l2
from .globalsettings import GlobalSettings
from . import performance as perf
from . import phcnn as phcnn


def _prepare_output_array(cv_k, cv_n, number_of_features, number_of_samples):
    total_number_iterations = cv_k * cv_n

    output = {
        'ranking': np.empty((total_number_iterations, number_of_features), dtype=np.int),
        'NPV': np.empty(total_number_iterations),
        'PPV': np.empty(total_number_iterations),
        'SENS': np.empty(total_number_iterations),
        'SPEC': np.empty(total_number_iterations),
        'MCC': np.empty(total_number_iterations),
        'AUC': np.empty(total_number_iterations),
        'DOR': np.empty(total_number_iterations),
        'ACC': np.empty(total_number_iterations),
        'ACCint': np.empty(total_number_iterations),
        'PREDS': np.zeros((total_number_iterations, number_of_samples), dtype=np.int),
        'REALPREDS_0': np.zeros((total_number_iterations, number_of_samples), dtype=np.float),
        'REALPREDS_1': np.zeros((total_number_iterations, number_of_samples), dtype=np.float)
    }

    for i in range(total_number_iterations):
        for j in range(number_of_samples):
            output['PREDS'][i][j] = -10
            output['REALPREDS_0'][i][j] = output['REALPREDS_1'][i][j] = -10.0

    return output


class _OptimizerNotFound(Exception):
    pass


def _configure_optimizer(selected_optimizer):
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
        # np.random.seed((n * CV_K) + i) #TODO: Find out why this was here
        np.random.shuffle(ranking)
    elif rank_method == 'ReliefF':
        relief = ReliefF(GlobalSettings.relief_k, seed=seed)
        relief.learn(xs_tr, ys_tr)
        w = relief.w()
        ranking = np.argsort(w)[::-1]
    elif rank_method == 'KBest':
        selector = SelectKBest(f_classif)
        selector.fit(xs_tr, ys_tr)
        ranking = np.argsort(-np.log10(selector.pvalues_))[::-1]
    else:
        raise _RankingNotFound("The only supported Ranking are random, ReliefF, KBest")

    return ranking


def _predict_classes(model, xs, coordinates, batch_size=32, verbose=1):
    '''
    Generate class predictions for the input samples batch by batch.
    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns
        A numpy array of class predictions.
    '''
    prb = model.predict({'xs_input': xs,
                         'coordinates_input': coordinates},
                        batch_size=batch_size, verbose=verbose)
    if prb.shape[-1] > 1:
        return prb.argmax(axis=-1)
    else:
        return (prb > 0.5).astype('int32')


def DAP(inputs):

    print('=' * 80)
    print('Building model')
    model = phcnn.PhcnnBuilder.build(
        nb_coordinates=inputs['nb_coordinates'],
        nb_features=inputs['nb_features'],
        nb_outputs=2
    )

    # opt = SGD(lr=0.001, nesterov=True, momentum=0.8, decay=1e-06)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())
    print('Model construction completed')

    output = _prepare_output_array(GlobalSettings.cv_k,
                                   GlobalSettings.cv_n,
                                   inputs['nb_features'],
                                   inputs['nb_samples'])

    for n in range(GlobalSettings.cv_n):
        idx = mlpy.cv_kfold(n=inputs['nb_samples'],
                            k=GlobalSettings.cv_k,
                            strat=inputs['ys'],
                            seed=n)
        print('=' * 80)
        print('{} over {} experiments'.format(n + 1, GlobalSettings.cv_n))

        for i, (idx_tr, idx_ts) in enumerate(idx):

            current = n * GlobalSettings.cv_k + i

            if not GlobalSettings.is_quiet:
                print('=' * 80)
                print('{} over {} folds'.format(i + 1, GlobalSettings.cv_k))

            xs_tr, xs_ts = inputs['xs'][idx_tr], inputs['xs'][idx_ts]
            ys_tr, ys_ts = inputs['ys'][idx_tr], inputs['ys'][idx_ts]
            crd_tr, crd_ts = inputs['coordinates'][idx_tr], inputs['coordinates'][idx_ts]

            ys_tr_cat = np_utils.to_categorical(ys_tr, inputs['nb_classes'])
            ys_ts_cat = np_utils.to_categorical(ys_ts, inputs['nb_classes'])

            crd_validation = inputs['coordinates'][0:inputs['validation_xs'].shape[0]]

            print('-- centering and normalization using: {}'.format(GlobalSettings.scaling))
            xs_tr, xs_ts = _apply_scaling(xs_tr, xs_ts, GlobalSettings.scaling)

            print('-- ranking the features using: {}'.format(GlobalSettings.rank_method))
            output['ranking'][current] = _get_ranking(xs_tr,
                                                      ys_tr,
                                                      GlobalSettings.rank_method,
                                                      seed=n)

            model.fit({'xs_input': xs_tr,
                       'coordinates_input': crd_tr},
                      {'output': ys_tr_cat},
                      epochs=20,
                      verbose=2,
                      validation_data=({'xs_input': inputs['validation_xs'],
                                        'coordinates_input': crd_validation},
                                       {'output': np_utils.to_categorical(inputs['validation_ys'])}
                                       )
                      )

            score, acc = model.evaluate({'xs_input': xs_ts,
                                         'coordinates_input': crd_ts},
                                        {'output': ys_ts_cat}, verbose=0)

            p = _predict_classes(model, xs_ts, crd_ts, verbose=0)
            pv = model.predict({'xs_input': xs_ts,
                                'coordinates_input': crd_ts}, verbose=0)
            pred_mcc = perf.KCCC_discrete(ys_ts, p)
            print('Test accuracy: {}'.format(acc))
            print('Test MCC: {}'.format(pred_mcc))

            output['PREDS'][current, idx_ts] = p
            output['REALPREDS_0'][current, idx_ts] = pv[:, 0]
            output['REALPREDS_1'][current, idx_ts] = pv[:, 1]

            output['NPV'][current] = perf.npv(ys_ts, p)
            output['PPV'][current] = perf.ppv(ys_ts, p)
            output['SENS'][current] = perf.sensitivity(ys_ts, p)
            output['SPEC'][current] = perf.specificity(ys_ts, p)
            output['MCC'][current] = perf.KCCC_discrete(ys_ts, p)
            output['AUC'][current] = roc_auc_score(ys_ts, p)
            output['DOR'][current] = perf.dor(ys_ts, p)
            output['ACC'][current] = perf.accuracy(ys_ts, p)
            output['ACCint'][current] = acc

    return output
