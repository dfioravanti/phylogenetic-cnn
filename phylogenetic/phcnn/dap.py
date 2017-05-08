import numpy as np
from keras.optimizers import Adam, RMSprop, SGD
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif

from phylogenetic.phcnn.relief import ReliefF
from phylogenetic.phcnn.scaling import norm_l2


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
        optimizer_configuration = {'name': rmsprop,
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


def _apply_scaling(xs_training, xs_test, selected_scaling):
    """
    Apply the scaling to the input data AFTER they have been divided in 
    (training, testing) during a cross validation. It raises an exception 
    if a wrong scaler name is required.
    :param xs_training: Training data
    :param xs_test: Test data
    :param selected_scaling: a string representing the select scaling
    :return: (xs_training, xs_test) scaled according to the scaler. 
    """
    if selected_scaling == 'norm_l2':
        xs_train_scaled, m_train, r_train = norm_l2(xs_training)
        xs_test_scaled, _, _ = norm_l2(xs_test, m_train, r_train)
    elif selected_scaling == 'std':
        scaler = preprocessing.StandardScaler(copy=False)
        xs_train_scaled = scaler.fit_transform(xs_training)
        xs_test_scaled = scaler.transform(xs_test)
    elif selected_scaling == 'minmax':
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        xs_train_scaled = scaler.fit_transform(xs_training)
        xs_test_scaled = scaler.transform(xs_test)
    elif selected_scaling == 'minmax0':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
        xs_train_scaled = scaler.fit_transform(xs_training)
        xs_test_scaled = scaler.transform(xs_test)
    else:
        raise _ScalingNotFound("The only supported scaling are norm_l2, std, minmax, minmax0")

    return xs_train_scaled, xs_test_scaled


class _RankingNotFound(Exception):
    pass


def _get_ranking(xs_training, ys_training, rank_method='ReliefF'):
    """
    Compute the ranking of the features as required. It raises an exception 
    if a wrong rank_method name is required.  
    
    :param xs_training: Training data of shape (number of training samples, number of features)
    :param ys_training: Training labels
    :param rank_method: a string representing the selected rank method
    :return: 
    """
    if rank_method == 'random':
        ranking = np.arange(xs_training.shape[1])
        # np.random.seed((n * CV_K) + i) #TODO: Find out why this was here
        np.random.shuffle(ranking)
    elif rank_method == 'ReliefF':
        relief = ReliefF(relief_k, seed=n)
        relief.learn(xs_training, ys_training)
        w = relief.w()
        ranking = np.argsort(w)[::-1]
    elif rank_method == 'KBest':
        selector = SelectKBest(f_classif)
        selector.fit(xs_training, ys_training)
        ranking = np.argsort(-np.log10(selector.pvalues_))[::-1]
    else:
        raise _RankingNotFound("The only supported Ranking are random, ReliefF, KBest")

    return ranking