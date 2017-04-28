## This code is written by Calogero Zarbo <zarbo@fbk.eu>.
## Based on code previously written by Davide Albanese, Marco Chierici and Alessandro Zandona'.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Requires Python >= 2.7, mlpy >= 3.5
from __future__ import division
#import matplotlib.pyplot as plt
import numpy as np
import csv
#from keras.utils.visualize_util import plot
import os.path
from scaling import norm_l2
import mlpy
from input_output import load_data
import performance as perf
import sys
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import ConfigParser
from distutils.version import StrictVersion
from collections import Counter
import tarfile
import glob
import pickle
import json
from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adam

from hyperas.distributions import choice, uniform, conditional

from keras.utils import np_utils


__version__ = '1.0'
__author__ = 'Albanese, Chierici, Zandona, Zarbo'

sys.setrecursionlimit(10000)
# def data():
#     import pickle
#     inputs = pickle.load(open('inputs.pickle', 'rb'))
#
#     SCALING =inputs['SCALING']
#     RANK_METHOD =inputs['RANK_METHOD']
#     OUTDIR_FSTEP= inputs['OUTDIR_FSTEP']
#     CV_N= inputs['CV_N']
#     CV_K= inputs['CV_K']
#     relief_k =inputs['relief_k']
#     rfe_p= inputs['rfe_p']
#     QUIET= inputs['QUIET']
#     x = inputs['x']
#     print x.shape
#     y = inputs['y']
#     return x, y, OUTDIR_FSTEP,SCALING,RANK_METHOD,CV_N,CV_K,relief_k,QUIET


def data():
    #import matplotlib.pyplot as plt
    #import matplotlib.pyplot as plt
    import keras
    #from keras.utils.visualize_util import plot
    import numpy as np
    import os.path
    from scaling import norm_l2
    import mlpy
    from input_output import load_data
    from sklearn import preprocessing
    from sklearn.metrics import roc_auc_score
    import ConfigParser
    from distutils.version import StrictVersion
    from collections import Counter
    import tarfile
    import glob
    import pickle
    import sys
    from hyperopt import Trials, STATUS_OK, tpe
    from hyperas import optim
    from hyperas.distributions import choice, uniform, conditional
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.callbacks import EarlyStopping
    import performance as perf
    import mlpy
    import numpy as np
    from keras.utils import np_utils
    from keras.models import model_from_config
    from keras.models import model_from_json
    from keras.optimizers import SGD, RMSprop, Adam
    from relief import ReliefF
    from sklearn.feature_selection import SelectKBest, f_classif
    import pickle
    input_file = 'inputs.pickle'
    return input_file

def model(input_file):
    #from keras.utils.visualize_util import plot
    import numpy as np
    import os.path
    from scaling import norm_l2
    import mlpy
    from input_output import load_data
    from sklearn import preprocessing
    from sklearn.metrics import roc_auc_score
    import ConfigParser
    from distutils.version import StrictVersion
    from collections import Counter
    import tarfile
    import glob
    import pickle
    import sys
    from hyperopt import Trials, STATUS_OK, tpe
    from hyperas import optim
    from hyperas.distributions import choice, uniform, conditional
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.callbacks import EarlyStopping
    import performance as perf
    import mlpy
    import numpy as np
    from keras.utils import np_utils
    from keras.models import model_from_config
    from keras.models import model_from_json
    from keras.optimizers import SGD, RMSprop, Adam
    from relief import ReliefF
    from sklearn.feature_selection import SelectKBest, f_classif
    import pickle
    import keras
    sys.setrecursionlimit(10000)

    inputs = pickle.load(open(input_file, 'rb'))

    SCALING =inputs['SCALING']
    RANK_METHOD =inputs['RANK_METHOD']
    OUTDIR_FSTEP= inputs['OUTDIR_FSTEP']
    CV_N= inputs['CV_N']
    CV_K= inputs['CV_K']
    relief_k =inputs['relief_k']
    rfe_p= inputs['rfe_p']
    QUIET= inputs['QUIET']
    x = inputs['x']
    y = inputs['y']

    n_classes = max(y)+1
    n_samples = x.shape[0]
    n_features = x.shape[1]

    # prepare output arrays
    RANKING = np.empty((CV_K*CV_N, n_features), dtype=np.int) #Finire la ricerca dell'NFEAT
    NPV = np.empty((CV_K*CV_N))
    PPV = np.empty_like(NPV)
    SENS = np.empty_like(NPV)
    SPEC = np.empty_like(NPV)
    MCC = np.empty_like(NPV)
    AUC = np.empty_like(NPV)
    DOR = np.empty_like(NPV)
    ACC = np.empty_like(NPV)
    hst_list = np.empty((CV_K*CV_N), dtype=list)
    ACCint = np.empty_like(NPV)



    PREDS = np.zeros((CV_K*CV_N, n_samples), dtype=np.int)
    for i in range(PREDS.shape[0]):
        for j in range(PREDS.shape[1]):
            PREDS[i][j] = -10

    REALPREDS_0 = np.zeros((CV_K*CV_N, n_samples), dtype=np.float)
    for i in range(REALPREDS_0.shape[0]):
        for j in range(REALPREDS_0.shape[1]):
            REALPREDS_0[i][j] = -10

    REALPREDS_1 = np.zeros((CV_K*CV_N, n_samples), dtype=np.float)
    for i in range(REALPREDS_1.shape[0]):
        for j in range(REALPREDS_1.shape[1]):
            REALPREDS_1[i][j] = -10

    trial_model = Sequential()
    trial_model.add(Dense({{choice([32,64, 128, 256, 512])}}, input_shape=(n_features,)))
    trial_model.add(Activation({{choice(['relu', 'sigmoid', 'tanh'])}}))
    trial_model.add(Dropout({{choice([0.25, 0.5, 0.75])}}))
    trial_model.add(Dense({{choice([32, 64, 128, 256, 512])}}))
    trial_model.add(Activation({{choice(['relu', 'sigmoid', 'tanh'])}}))
    trial_model.add(Dropout({{choice([0.25, 0.5, 0.75])}}))

    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        trial_model.add(Dense({{choice([32, 64, 128, 256, 512])}}))
        trial_model.add(Activation({{choice(['relu', 'sigmoid', 'tanh'])}}))
        trial_model.add(Dropout({{choice([0.25, 0.5, 0.75])}}))

    trial_model.add(Dense(n_classes))
    trial_model.add(Activation('softmax'))
    if n_classes > 2:
        trial_model_loss = 'categorical_crossentropy'
    else:
        trial_model_loss = 'binary_crossentropy'
    trial_model_optimizer_list = {{choice(['rmsprop', 'adam', 'sgd'])}}
    trial_model_optimizer_dict = {}
    print "#Chosen Optimizer: ", trial_model_optimizer_list
    if trial_model_optimizer_list == 'adam':
        epsilon = 1e-08
        lr = {{choice([0.1, 0.01 , 0.001, 0.0001])}}
        beta_1 = 0.9
        beta_2 = 0.999
        trial_model_optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,epsilon=epsilon )
        trial_model_optimizer_dict['adam'] = {'lr': lr,
                                              'beta_1': beta_1,
                                              'beta_2': beta_2,
                                              'epsilon': epsilon}
    elif trial_model_optimizer_list == 'rmsprop':
        epsilon = 1e-08
        lr = {{choice([0.1, 0.01 , 0.001, 0.0001])}}
        rho = 0.9
        trial_model_optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon)
        trial_model_optimizer_dict['rmsprop'] = {'lr': lr,
                                              'rho': rho,
                                              'epsilon': epsilon}
    elif trial_model_optimizer_list == 'sgd':
        nesterov = {{choice([True,False])}}
        lr = {{choice([0.1, 0.01 , 0.001, 0.0001])}}
        momentum = 0.9
        decay = 1e-06
        trial_model_optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
        trial_model_optimizer_dict['sgd'] = {'lr': lr,
                                              'momentum': momentum,
                                              'decay': decay,
                                              'nesterov': nesterov}
    trial_model_batch_size = {{choice([4, 8, 16])}}

    saved_clean_model = trial_model.to_json()

    for n in range(CV_N):
        idx = mlpy.cv_kfold(n=x.shape[0], k=CV_K, strat=y, seed=n)
        print "=" * 80
        print "%d over %d experiments" % (n+1, CV_N)

        for i, (idx_tr, idx_ts) in enumerate(idx):

            if not QUIET:
                print "_" * 80
                print "-- %d over %d folds" % (i+1, CV_K)

            x_tr, x_ts = x[idx_tr], x[idx_ts]
            y_tr, y_ts = y[idx_tr], y[idx_ts]


            y_tr_cat = np_utils.to_categorical(y_tr, n_classes)
            y_ts_cat = np_utils.to_categorical(y_ts, n_classes)


            # centering and normalization
            print "-- centering and normalization:",SCALING
            if SCALING == 'norm_l2':
                x_tr, m_tr, r_tr = norm_l2(x_tr)
                x_ts, _, _ = norm_l2(x_ts, m_tr, r_tr)
            elif SCALING == 'std':
                scaler = preprocessing.StandardScaler(copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)
            elif SCALING == 'minmax':
                scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)
            elif SCALING == 'minmax0':
                scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)

            print "-- ranking the features:",RANK_METHOD
            if RANK_METHOD == 'random':
                ranking_tmp = np.arange(n_features)
                np.random.seed((n * CV_K) + i)
                np.random.shuffle(ranking_tmp)
            elif RANK_METHOD == 'ReliefF':
                relief = ReliefF(relief_k, seed=n)
                relief.learn(x_tr, y_tr)
                w = relief.w()
                ranking_tmp = np.argsort(w)[::-1]
            elif RANK_METHOD == 'KBest':
                selector = SelectKBest(f_classif)
                selector.fit(x_tr, y_tr)
                ranking_tmp = np.argsort(-np.log10(selector.pvalues_))[::-1]

            RANKING[(n * CV_K) + i] = ranking_tmp

            print '-- real labels model evaluation'
            trial_model = model_from_json(saved_clean_model)
            print '##Compiling'
            trial_model.compile(loss=trial_model_loss, optimizer=trial_model_optimizer)
            print '##Fitting'
            #filepath = 'weights.best-model_CV-N_'+str(n)+'_CV-K_'+str(i)+'.hdf5'

            hst_list[(n * CV_K) + i]=trial_model.fit(x_tr, y_tr_cat,
                  batch_size=trial_model_batch_size,
                  nb_epoch=200,
                  # show_accuracy=True,
                  verbose=0)
                  #callbacks=[keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')])


            print "##Evaluating"
            acc = trial_model.evaluate(x_ts, y_ts_cat, verbose=0)
            # print "Acc:", acc, "Type:",type(acc)
            print "##Getting Labels on Unseen Data"
            p = trial_model.predict_classes(x_ts, verbose=0)
            pv = trial_model.predict_proba(x_ts,verbose=0)
            pred_mcc = perf.KCCC_discrete(y_ts, p)
            print '\tTest accuracy:', acc
            print '\tTest MCC:', pred_mcc

            PREDS[(n * CV_K + i), idx_ts] = p
            REALPREDS_0[(n * CV_K + i), idx_ts] = pv[:, 0]
            REALPREDS_1[(n * CV_K + i), idx_ts] = pv[:, 1]

            NPV[(n * CV_K) + i] = perf.npv(y_ts, p)
            PPV[(n * CV_K) + i] = perf.ppv(y_ts, p)
            SENS[(n * CV_K) + i] = perf.sensitivity(y_ts, p)
            SPEC[(n * CV_K) + i] = perf.specificity(y_ts, p)
            MCC[(n * CV_K) + i] = perf.KCCC_discrete(y_ts, p)
            AUC[(n * CV_K) + i] = roc_auc_score(y_ts, p)
            DOR[(n * CV_K) + i] = perf.dor(y_ts, p)
            ACC[(n * CV_K) + i] = perf.accuracy(y_ts, p)
            ACCint[(n * CV_K) + i] = acc


    trial_package = {
        'cv_history': hst_list,
        'model': saved_clean_model,
        'loss': trial_model_loss,
        'optimizer': trial_model_optimizer_dict,
        'RANKING': RANKING,
        'batch_size': trial_model_batch_size,
        'true_labels':{
            'PREDS':PREDS,
            'REALPREDS_0':REALPREDS_0,
            'REALPREDS_1':REALPREDS_1,
            'NPV':NPV,
            'PPV':PPV,
            'SENS':SENS,
            'SPEC':SPEC,
            'MCC':MCC,
            'AUC':AUC,
            'DOR':DOR,
            'ACC':ACC,
            'ACCint':ACCint
        }

    }

    #keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
    #return {'loss': -np.mean(ACCint), 'status': STATUS_OK, 'model': trial_package}
    #return {'loss': -(np.mean(MCC)-abs(np.mean(MCC_random))), 'status': STATUS_OK, 'model': trial_package}
    return {'loss': -np.mean(MCC), 'status': STATUS_OK, 'model': trial_package}

class myArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(myArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        for arg in line.split():
            if not arg.strip():
                continue
            if arg[0] == '#':
                break
            yield arg

parser = myArgumentParser(description='Run a training experiment (10x5-CV fold) using Rectified Factor Networks, Support Vector Machines, Random Forests and/or Multilayer Perceptron.',
        fromfile_prefix_chars='@')
parser.add_argument('DATAFILE', type=str, help='Training datafile')
parser.add_argument('LABELSFILE', type=str, help='Sample labels')
parser.add_argument('OUTDIR', type=str, help='Output directory')
parser.add_argument('--scaling', dest='SCALING', type=str, choices=['norm_l2', 'std', 'minmax', 'minmax0'], default='std', help='Scaling method (default: %(default)s)')
parser.add_argument('--ranking', dest='RANK_METHOD', type=str, choices=['ReliefF', 'tree', 'KBest', 'random'], default='ReliefF', help='Feature ranking method: ReliefF, extraTrees, Anova F-score, random ranking (default: %(default)s)')
parser.add_argument('--cv_k', type=np.int, default=5, help='Number of CV folds (default: %(default)s)')
parser.add_argument('--cv_n', type=np.int, default=10, help='Number of CV cycles (default: %(default)s)')
parser.add_argument('--reliefk', type=np.int, default=3, help='Number of nearest neighbors for ReliefF (default: %(default)s)')
parser.add_argument('--rfep', type=np.float, default=0.2, help='Fraction of features to remove at each iteration in RFE (p=0 one variable at each step, p=1 naive ranking) (default: %(default)s)')
parser.add_argument('--plot', action='store_true', help='Plot metric values over all training cycles' )
parser.add_argument('--tsfile', type=str, default=None, help='Validation datafile')
parser.add_argument('--tslab', type=str, default=None, help='Validation labels, if available')
parser.add_argument('--trials', type=int, default=None, help='Number of hypersearch trials.')
parser.add_argument('--quiet', action='store_true', help='Run quietly (no progress info)')
parser.add_argument('--allfeatures', action='store_true', help='Do not perform features step')

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
DATAFILE = args.DATAFILE
LABELSFILE = args.LABELSFILE
SCALING = args.SCALING
RANK_METHOD = args.RANK_METHOD
OUTDIR = args.OUTDIR
plot_out = args.plot
CV_N = args.cv_n
CV_K = args.cv_k
relief_k = args.reliefk
rfe_p = args.rfep
QUIET = args.quiet
TESTFILE = args.tsfile
TSLABELSFILE = args.tslab
TRIALS = args.trials


try:
    os.makedirs(OUTDIR)
except OSError:
    if not os.path.isdir(OUTDIR):
        raise


sample_names, var_names, x = load_data(DATAFILE)
y = np.loadtxt(LABELSFILE, dtype=np.int)

mcc_cv_val = [['N_FEATURES', 'MCC_CV', 'MCC_VAL', 'MCC_TrainOnVal']]


while len(var_names) >= 5:
    print "Preparing Model Selection with", len(var_names), "Features"

    OUTDIR_FSTEP = OUTDIR+"/"+str(len(var_names))+"_Features/"
    try:
        os.makedirs(OUTDIR_FSTEP)
    except OSError:
        if not os.path.isdir(OUTDIR_FSTEP):
            raise

    inputs = {
        'sample_names' : sample_names,
        'var_names' : var_names[:],
        'x':np.copy(x),
        'y':np.copy(y),
        'SCALING' : SCALING,
        'RANK_METHOD' : RANK_METHOD,
        'OUTDIR_FSTEP' : OUTDIR_FSTEP,
        'plot_out' : plot_out,
        'CV_N' : CV_N,
        'CV_K' : CV_K,
        'relief_k' : relief_k,
        'rfe_p' : rfe_p,
        'QUIET' : QUIET
    }

    with open('inputs.pickle', 'wb') as handle:
        pickle.dump(inputs, handle)

    best_run, best_model = optim.minimize(model=model,
                                              data=data,
                                              algo=tpe.suggest,
                                              max_evals=TRIALS,
                                              trials=Trials())

    #prepare output files
    RANKING = best_model['RANKING']
    PREDS = best_model['true_labels']['PREDS']
    REALPREDS_0 = best_model['true_labels']['REALPREDS_0']
    REALPREDS_1 = best_model['true_labels']['REALPREDS_1']
    LOSS = best_model['loss']
    OPTIMIZER_dict = best_model['optimizer']

    if OPTIMIZER_dict.keys()[0] == 'adam':
        OPTIMIZER = Adam(lr=OPTIMIZER_dict['adam']['lr'], beta_1=OPTIMIZER_dict['adam']['beta_1'], beta_2=OPTIMIZER_dict['adam']['beta_2'],epsilon=OPTIMIZER_dict['adam']['epsilon'])
    elif OPTIMIZER_dict.keys()[0] == 'rmsprop':
        OPTIMIZER = RMSprop(lr=OPTIMIZER_dict['rmsprop']['lr'], rho=OPTIMIZER_dict['rmsprop']['rho'], epsilon=OPTIMIZER_dict['rmsprop']['epsilon'])
    elif OPTIMIZER_dict.keys()[0] == 'sgd':
        OPTIMIZER = SGD(lr=OPTIMIZER_dict['sgd']['lr'], momentum=OPTIMIZER_dict['sgd']['momentum'], decay=OPTIMIZER_dict['sgd']['decay'], nesterov=OPTIMIZER_dict['sgd']['nesterov'])

    n_features = len(var_names)
    BASEFILE = os.path.splitext(os.path.basename(DATAFILE))[0]
    OUTFILE = os.path.join(OUTDIR_FSTEP, '_'.join([BASEFILE, "DNN", RANK_METHOD, SCALING]))

    #saving model
    best_model_parameters = {'model': best_model['model'],
                             'loss': LOSS,
                             'optimizer': OPTIMIZER_dict,
                             'batch_size': best_model['batch_size']}
    with open(os.path.realpath( OUTFILE + "_model.json" ), 'w') as out_model:
        json.dump(best_model_parameters, out_model)

    historyf = open(OUTFILE + "_history.txt", 'w')
    history_w = csv.writer(historyf, delimiter='\t', lineterminator='\n')

    metricsf = open(OUTFILE + "_metrics.txt", 'w')
    metrics_w = csv.writer(metricsf, delimiter='\t', lineterminator='\n')

    metricsf_random = open(OUTFILE + "_metricsrandom.txt", 'w')
    metrics_w_random = csv.writer(metricsf_random, delimiter='\t', lineterminator='\n')

    rankingf = open(OUTFILE + "_featurelist.txt", 'w')
    ranking_w = csv.writer(rankingf, delimiter='\t', lineterminator='\n')
    ranking_w.writerow(["FEATURE_ID", "FEATURE_NAME", "MEAN_POS", "MEDIAN_ALL", "MEDIAN_0", "MEDIAN_1", "FOLD_CHANGE", "LOG2_FOLD_CHANGE"])

    stabilityf = open(OUTFILE + "_stability.txt", 'w')
    stability_w = csv.writer(stabilityf, delimiter='\t', lineterminator='\n')

    predf = open(OUTFILE + "_preds.txt", 'w')
    pred_w = csv.writer(predf, delimiter='\t', lineterminator='\n')


    pred_w.writerow(sample_names)
    for row in PREDS:
        pred_w.writerow(row.tolist())
    predf.close()

    real_lbl = {}

    for i in range(len(sample_names)):
        if y[i] == -1:
            y[i] = 0
        real_lbl[sample_names[i]] = y[i]

    pred_lbl = {}

    for line in PREDS:
        for i in range(len(line)):
            if sample_names[i] not in pred_lbl.keys():
                pred_lbl[sample_names[i]] = [0,0]
            if line[i] != '-99':
                if line[i] == real_lbl[sample_names[i]]:
                    pred_lbl[sample_names[i]][0] += 1
                pred_lbl[sample_names[i]][1] += 1

    pred_performance_f = open(OUTFILE + "_preds_performance.txt", 'w')
    pred_performance_w = csv.writer(pred_performance_f, delimiter='\t', lineterminator='\n')
    pred_performance_w.writerow(["SAMPLE_NAME", "CORRECT_RATE", "REAL_LABEL"])
    for k,v in pred_lbl.items():
        rate = float(v[0])/float(v[1])
        pred_performance_w.writerow([k, rate, real_lbl[k]])

    pred_performance_f.close()



    realpredf = open(OUTFILE + "_realpreds_0.txt", 'w')
    realpred_w = csv.writer(realpredf, delimiter='\t', lineterminator='\n')
    realpred_w.writerow(sample_names)
    for row in REALPREDS_0:
        realpred_w.writerow(row.tolist())

    realpredf.close()


    realpredf = open(OUTFILE + "_realpreds_1.txt", 'w')
    realpred_w = csv.writer(realpredf, delimiter='\t', lineterminator='\n')
    realpred_w.writerow(sample_names)
    for row in REALPREDS_1:
        realpred_w.writerow(row.tolist())

    realpredf.close()

    # write metrics for all CV iterations
    history = best_model['cv_history']
    MCC = best_model['true_labels']['MCC']
    SENS = best_model['true_labels']['SENS']
    SPEC = best_model['true_labels']['SPEC']
    PPV = best_model['true_labels']['PPV']
    NPV = best_model['true_labels']['NPV']
    AUC = best_model['true_labels']['AUC']
    ACC = best_model['true_labels']['ACC']
    DOR = best_model['true_labels']['DOR']

    for fold in history:
        history_w.writerow(fold.history['loss'])

    np.savetxt(OUTFILE + "_allmetrics_MCC.txt", MCC, fmt='%.4f', delimiter='\t')
    np.savetxt(OUTFILE + "_allmetrics_SENS.txt", SENS, fmt='%.4f', delimiter='\t')
    np.savetxt(OUTFILE + "_allmetrics_SPEC.txt", SPEC, fmt='%.4f', delimiter='\t')
    np.savetxt(OUTFILE + "_allmetrics_PPV.txt", PPV, fmt='%.4f', delimiter='\t')
    np.savetxt(OUTFILE + "_allmetrics_NPV.txt", NPV, fmt='%.4f', delimiter='\t')
    np.savetxt(OUTFILE + "_allmetrics_AUC.txt", AUC, fmt='%.4f', delimiter='\t')
    np.savetxt(OUTFILE + "_allmetrics_ACC.txt", ACC, fmt='%.4f', delimiter='\t')
    np.savetxt(OUTFILE + "_allmetrics_DOR.txt", DOR, fmt='%.4f', delimiter='\t')

    tar = tarfile.open(OUTFILE + "_allmetrics.tar.gz", "w:gz")
    for metricFile in glob.glob(OUTFILE + "_allmetrics_*txt"):
        tar.add(metricFile, arcname=os.path.basename(metricFile))
        os.unlink(metricFile)
    tar.close()

    # write all rankings
    np.savetxt(OUTFILE + "_ranking.csv.gz", RANKING, fmt='%d', delimiter='\t')

    # average values

    AMCC = np.mean(MCC, axis=0)
    ASENS = np.mean(SENS, axis=0)
    ASPEC = np.mean(SPEC, axis=0)
    APPV = np.mean(PPV, axis=0)
    ANPV = np.mean(NPV, axis=0)
    AAUC = np.mean(AUC, axis=0)
    AACC = np.mean(ACC, axis=0)
    ADOR = np.mean(DOR, axis=0)
    # approximated Odds Ratio, computed from ASENS and ASPEC (to avoid inf and nan values)
    ADOR_APPROX = (ASENS / (1 - ASPEC)) / ((1 - ASENS) / ASPEC)

    # confidence intervals
    NPVCI = mlpy.bootstrap_ci(NPV)
    PPVCI = mlpy.bootstrap_ci(PPV)
    SENSCI = mlpy.bootstrap_ci(SENS)
    SPECCI = mlpy.bootstrap_ci(SPEC)
    MCCCI = mlpy.bootstrap_ci(MCC)
    AUCCI = mlpy.bootstrap_ci(AUC)
    DORCI = mlpy.bootstrap_ci(DOR)
    ACCCI = mlpy.bootstrap_ci(ACC)

    # Borda list
    BORDA_ID, _, BORDA_POS = mlpy.borda_count(RANKING)

    # Canberra stability indicator
    PR = np.argsort( RANKING )
    STABILITY = mlpy.canberra_stability(PR, n_features)

    metrics_w.writerow(["STEP",
                        "MCC", "MCC_MIN", "MCC_MAX",
                        "SENS", "SENS_MIN", "SENS_MAX",
                        "SPEC", "SPEC_MIN", "SPEC_MAX",
                        "PPV", "PPV_MIN", "PPV_MAX",
                        "NPV", "NPV_MIN", "NPV_MAX",
                        "AUC", "AUC_MIN", "AUC_MAX",
                        "ACC", "ACC_MIN", "ACC_MAX",
                        "DOR", "DOR_MIN", "DOR_MAX",
                        "DOR_APPROX"])

    stability_w.writerow(["STEP", "STABILITY"])


    metrics_w.writerow([n_features,
                        AMCC, MCCCI[0], MCCCI[1],
                        ASENS, SENSCI[0], SENSCI[1],
                        ASPEC, SPECCI[0], SPECCI[1],
                        APPV, PPVCI[0], PPVCI[1],
                        ANPV, NPVCI[0], NPVCI[1],
                        AAUC, AUCCI[0], AUCCI[1],
                        AACC, ACCCI[0], ACCCI[1],
                        ADOR, DORCI[0], DORCI[1],
                        ADOR_APPROX])
    stability_w.writerow([n_features, STABILITY])

    metricsf.close()
    stabilityf.close()




    print "========########## Starting Random label ############==================="

    NPV_random = np.empty((CV_K*CV_N))
    PPV_random = np.empty_like(NPV)
    SENS_random = np.empty_like(NPV)
    SPEC_random = np.empty_like(NPV)
    MCC_random = np.empty_like(NPV)
    AUC_random = np.empty_like(NPV)
    DOR_random = np.empty_like(NPV)
    ACC_random = np.empty_like(NPV)
    ACCint_random = np.empty_like(NPV)

    n_classes = max(y)+1
    n_samples = x.shape[0]
    n_features = x.shape[1]

    ys = np.copy(y)
    np.random.seed(0)
    np.random.shuffle(ys)

    loss_per_fold = []

    for n in range(CV_N):
        idx = mlpy.cv_kfold(n=x.shape[0], k=CV_K, strat=ys, seed=n)
        print "=" * 80
        print "%d over %d experiments" % (n+1, CV_N)

        for i, (idx_tr, idx_ts) in enumerate(idx):

            if not QUIET:
                print "_" * 80
                print "-- %d over %d folds" % (i+1, CV_K)

            x_tr, x_ts = x[idx_tr], x[idx_ts]
            y_tr, y_ts = ys[idx_tr], ys[idx_ts]

            y_tr_cat = np_utils.to_categorical(y_tr, n_classes)
            y_ts_cat = np_utils.to_categorical(y_ts, n_classes)

            # centering and normalization
            print "-- centering and normalization:",SCALING
            if SCALING == 'norm_l2':
                x_tr, m_tr, r_tr = norm_l2(x_tr)
                x_ts, _, _ = norm_l2(x_ts, m_tr, r_tr)
            elif SCALING == 'std':
                scaler = preprocessing.StandardScaler(copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)
            elif SCALING == 'minmax':
                scaler = preprocessing.MinMaxScaler(feature_range=(-1,1), copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)
            elif SCALING == 'minmax0':
                scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=False)
                x_tr = scaler.fit_transform(x_tr)
                x_ts = scaler.transform(x_ts)


            print '-- random labels model evaluation'
            trial_model = model_from_json(best_model_parameters['model'])
            trial_model.compile(loss=LOSS, optimizer=OPTIMIZER)
            h =trial_model.fit(x_tr, y_tr_cat,
                  batch_size=best_model['batch_size'],
                  nb_epoch=200,
                  verbose=0)

            acc = trial_model.evaluate(x_ts, y_ts_cat, verbose=0)
            p = trial_model.predict_classes(x_ts, verbose=0)
            # pv = trial_model.predict_proba(x_ts,verbose=0)
            pred_mcc = perf.KCCC_discrete(y_ts, p)
            print '\tTest accuracy:', acc
            print '\tTest MCC:', pred_mcc

            NPV_random[(n * CV_K) + i] = perf.npv(y_ts, p)
            PPV_random[(n * CV_K) + i] = perf.ppv(y_ts, p)
            SENS_random[(n * CV_K) + i] = perf.sensitivity(y_ts, p)
            SPEC_random[(n * CV_K) + i] = perf.specificity(y_ts, p)
            MCC_random[(n * CV_K) + i] = perf.KCCC_discrete(y_ts, p)
            AUC_random[(n * CV_K) + i] = roc_auc_score(y_ts, p)
            DOR_random[(n * CV_K) + i] = perf.dor(y_ts, p)
            ACC_random[(n * CV_K) + i] = perf.accuracy(y_ts, p)
            ACCint_random[(n * CV_K) + i] = acc

            loss_per_fold.append(h.history)

    #summarize history for loss
    #plt.plot(h.history['loss'])
    #plt.plot(h.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #pylab.savefig('.png')

    # average values random

    AMCC_random = np.mean(MCC_random, axis=0)
    ASENS_random = np.mean(SENS_random, axis=0)
    ASPEC_random = np.mean(SPEC_random, axis=0)
    APPV_random = np.mean(PPV_random, axis=0)
    ANPV_random = np.mean(NPV_random, axis=0)
    AAUC_random = np.mean(AUC_random, axis=0)
    AACC_random = np.mean(ACC_random, axis=0)
    ADOR_random = np.mean(DOR_random, axis=0)
    # approximated Odds Ratio, computed from ASENS and ASPEC (to avoid inf and nan values)
    ADOR_APPROX_random = (ASENS_random / (1 - ASPEC_random)) / ((1 - ASENS_random) / ASPEC_random)

    # confidence intervals
    NPVCI_random = mlpy.bootstrap_ci(NPV_random)
    PPVCI_random = mlpy.bootstrap_ci(PPV_random)
    SENSCI_random = mlpy.bootstrap_ci(SENS_random)
    SPECCI_random = mlpy.bootstrap_ci(SPEC_random)
    MCCCI_random = mlpy.bootstrap_ci(MCC_random)
    AUCCI_random = mlpy.bootstrap_ci(AUC_random)
    DORCI_random = mlpy.bootstrap_ci(DOR_random)
    ACCCI_random = mlpy.bootstrap_ci(ACC_random)

    metrics_w_random.writerow(["STEP",
                        "MCC", "MCC_MIN", "MCC_MAX",
                        "SENS", "SENS_MIN", "SENS_MAX",
                        "SPEC", "SPEC_MIN", "SPEC_MAX",
                        "PPV", "PPV_MIN", "PPV_MAX",
                        "NPV", "NPV_MIN", "NPV_MAX",
                        "AUC", "AUC_MIN", "AUC_MAX",
                        "ACC", "ACC_MIN", "ACC_MAX",
                        "DOR", "DOR_MIN", "DOR_MAX",
                        "DOR_APPROX"])

    metrics_w_random.writerow([n_features,
                               AMCC_random, MCCCI_random[0], MCCCI_random[1],
                               ASENS_random, SENSCI_random[0], SENSCI_random[1],
                               ASPEC_random, SPECCI_random[0], SPECCI_random[1],
                               APPV_random, PPVCI_random[0], PPVCI_random[1],
                               ANPV_random, NPVCI_random[0], NPVCI_random[1],
                               AAUC_random, AUCCI_random[0], AUCCI_random[1],
                               AACC_random, ACCCI_random[0], ACCCI_random[1],
                               ADOR_random, DORCI_random[0], DORCI_random[1],
                               ADOR_APPROX_random])

    metricsf_random.close()


    for i, pos in zip(BORDA_ID, BORDA_POS):
        classes = np.unique(y)
        med_all = np.median(x[:, i])
        med_c = np.zeros(np.shape(classes)[0])
        for jj,c in enumerate(classes):
            med_c[jj] = np.median(x[y==c, i])
        with np.errstate(divide='ignore'):
            fc = med_c[1] / med_c[0]
        log2fc = np.log2(fc)
        ranking_w.writerow([ i, var_names[i], pos+1, med_all, med_c[0], med_c[1], fc, log2fc ])
    rankingf.close()


    logf = open(OUTFILE + ".log", 'w')
    config = ConfigParser.RawConfigParser()
    config.add_section("SOFTWARE VERSIONS")
    config.set("SOFTWARE VERSIONS", os.path.basename(__file__), __version__)
    config.set("SOFTWARE VERSIONS", "Python", sys.version.replace('\n', ''))
    config.set("SOFTWARE VERSIONS", "Numpy", np.__version__)
    config.set("SOFTWARE VERSIONS", "MLPY", mlpy.__version__)
    config.add_section("CV PARAMETERS")
    config.set("CV PARAMETERS", "Folds", CV_K)
    config.set("CV PARAMETERS", "Iterations", CV_N)
    config.add_section("INPUT")
    config.set("INPUT", "Data", os.path.realpath( DATAFILE ))
    config.set("INPUT", "Labels", os.path.realpath( LABELSFILE ))
    config.set("INPUT", "Scaling", SCALING)
    config.set("INPUT", "Rank_method", RANK_METHOD)
    config.set("INPUT", "Model", os.path.realpath( OUTFILE + "_model.json" ))
    config.add_section("OUTPUT")
    config.set("OUTPUT", "Metrics", os.path.realpath( OUTFILE + "_metrics.txt" ))
    config.set("OUTPUT", "Borda", os.path.realpath( OUTFILE + "_featurelist.txt" ))
    config.set("OUTPUT", "Internal", os.path.realpath( OUTFILE + "_internal.txt" ))
    config.set("OUTPUT", "Stability", os.path.realpath( OUTFILE + "_stability.txt" ))
    config.set("OUTPUT", "MCC", np.max(AMCC))
    config.write(logf)
    logf.close()

    #####################################
    ### SAVING THE FULL TRAINED MODEL ###
    #####################################
    x_tr = np.copy(x)
    OUTDIR_FSTEP_FULL_TRAINED_MODEL = OUTDIR + "/" + str(len(var_names)) + "_Features/FULL_TRAINED_MODEL/"
    try:
        os.makedirs(OUTDIR_FSTEP_FULL_TRAINED_MODEL)
    except OSError:
        if not os.path.isdir(OUTDIR_FSTEP_FULL_TRAINED_MODEL):
            raise

    idx = []
    y_tr_cat = np_utils.to_categorical(y, np.max(y) + 1)

    # centering and normalization
    if SCALING == 'norm_l2':
        x_tr, m_tr, r_tr = norm_l2(x_tr)
    elif SCALING == 'std':
        scaler = preprocessing.StandardScaler(copy=False)
        x_tr = scaler.fit_transform(x_tr)
    elif SCALING == 'minmax':
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        x_tr = scaler.fit_transform(x_tr)
    elif SCALING == 'minmax0':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
        x_tr = scaler.fit_transform(x_tr)

    validation_model = model_from_json(best_model_parameters['model'])
    validation_model.compile(loss=LOSS, optimizer=OPTIMIZER)
    val_hist = validation_model.fit(x_tr, y_tr_cat,
                         batch_size=best_model['batch_size'],
                         nb_epoch=200,
                         show_accuracy=True,
                         verbose=0)
    #plot(validation_model, to_file='model.png')


    with open(OUTDIR_FSTEP_FULL_TRAINED_MODEL + "FULL_TRAINED_MODEL.pickle", 'wb') as handle:
        pickle.dump(validation_model, handle)


    ################
    ###VALIDATION###
    ################
    if TESTFILE is not None:
        x_tr = np.copy(x)
        OUTDIR_FSTEP_VALIDATION = OUTDIR+"/"+str(len(var_names))+"_Features/VALIDATION/"
        try:
            os.makedirs(OUTDIR_FSTEP_VALIDATION)
        except OSError:
            if not os.path.isdir(OUTDIR_FSTEP_VALIDATION):
                raise
        BASEFILE_VALIDATION = os.path.splitext(os.path.basename(TESTFILE))[0]
        OUTFILE_VALIDATION = os.path.join(OUTDIR_FSTEP_VALIDATION, '_'.join([BASEFILE_VALIDATION, "DNN", RANK_METHOD, SCALING]))
        print 'valid', TESTFILE, len(var_names)
        valid_historyf = open(OUTFILE_VALIDATION + "_history.txt", 'w')
        valid_history_w = csv.writer(valid_historyf, delimiter='\t', lineterminator='\n')
        valid_history_w.writerow(val_hist.history['loss'])

        sample_names_ts, var_names_ts, x_ts = load_data(TESTFILE)
        # load the TS labels if available
        if TSLABELSFILE is not None:
            y_ts = np.loadtxt(TSLABELSFILE, dtype=np.int, delimiter='\t')

        idx = []
        for i in range(0, len(var_names)):
            if var_names[i] in var_names_ts:
                idx.append(var_names_ts.index(var_names[i]))
            else:
                print var_names[i]
        # considering samples names in the new table
        x_ts = x_ts[:, idx]

        y_tr_cat = np_utils.to_categorical(y, np.max(y)+1)

        # centering and normalization
        if SCALING == 'norm_l2':
            x_tr, m_tr, r_tr = norm_l2(x_tr)
            x_ts, _, _ = norm_l2(x_ts, m_tr, r_tr)
        elif SCALING == 'std':
            scaler = preprocessing.StandardScaler(copy=False)
            x_tr = scaler.fit_transform(x_tr)
            x_ts = scaler.transform(x_ts)
        elif SCALING == 'minmax':
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
            x_tr = scaler.fit_transform(x_tr)
            x_ts = scaler.transform(x_ts)
        elif SCALING == 'minmax0':
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=False)
            x_tr = scaler.fit_transform(x_tr)
            x_ts = scaler.transform(x_ts)

        p_tr = validation_model.predict_classes(x_tr, batch_size=best_model['batch_size'], verbose=0)
        p_ts = validation_model.predict_classes(x_ts, batch_size=best_model['batch_size'], verbose=0)

        prob_tr = validation_model.predict_proba(x_tr, batch_size=best_model['batch_size'], verbose=0)
        prob_ts = validation_model.predict_proba(x_ts, batch_size=best_model['batch_size'], verbose=0)

        mcc_train_on_val = perf.KCCC_discrete(y, p_tr)
        print "MCC on train: %.3f" % mcc_train_on_val
        if TSLABELSFILE is not None:
            mcc_val = perf.KCCC_discrete(y_ts, p_ts)
            print "MCC on validation: %.3f" % mcc_val
            mcc_cv_val.append([str(len(var_names)), str(AMCC), str(mcc_val), str(mcc_train_on_val)])

        # write output files
        fout = open(OUTFILE_VALIDATION + "_TEST_MCC.txt", "w")
        fout.write("MCC on train: %.3f" % mcc_train_on_val+"\n")


        if TSLABELSFILE is not None:
            fout.write("MCC on validation: %.3f" % mcc_val+"\n")
        fout.close()

        fout = open(OUTFILE_VALIDATION + "_TEST_pred_tr.txt", "w")
        for i in range(len(sample_names)):
            fout.write("%s\t%i\n" % (sample_names[i], p_tr[i]))
        fout.close()

        fout = open(OUTFILE_VALIDATION + "_TEST_pred_ts.txt", "w")
        for i in range(len(sample_names_ts)):
            fout.write("%s\t%i\n" % (sample_names_ts[i], p_ts[i]))
        fout.close()

        np.savetxt(OUTFILE_VALIDATION + "_TEST_signature.txt",
                   np.array(var_names).reshape(-1,1),
                   fmt='%s', delimiter='\t')

        fout = open(OUTFILE_VALIDATION + "_TEST_prob_tr.txt", "w")
        fout.write("SAMPLE\tCLASS 0\tCLASS 1\n")
        for i in range(len(sample_names)):
            fout.write("%s\t%f\t%f\n" % (sample_names[i], prob_tr[i,0], prob_tr[i,1]))
        fout.close()

        fout = open(OUTFILE_VALIDATION + "_TEST_prob_ts.txt", "w")
        fout.write("SAMPLE\tCLASS 0\tCLASS 1\n")
        for i in range(len(sample_names_ts)):
            fout.write("%s\t%f\t%f\n" % (sample_names_ts[i], prob_ts[i,0], prob_ts[i,1]))
        fout.close()
    valid_historyf = open(OUTFILE_VALIDATION + "_history.txt", 'w')
    valid_history_w = csv.writer(valid_historyf, delimiter='\t', lineterminator='\n')
    valid_history_w.writerow(val_hist.history['loss'])
    ####################
    ###VALIDATION END###
    ####################

    rank = np.loadtxt(OUTFILE + "_featurelist.txt", delimiter = '\t', skiprows = 1, dtype = str)
    feats = rank[:, 1]
    nfeat = int(np.ceil(len(var_names)/100 * 80))
    top_feats = feats[0:nfeat]

    # extract top features from table with abundances of all features
    idx = []
    for i in range(0, nfeat):
        if top_feats[i] in var_names:
            idx.append(var_names.index(top_feats[i]))
        else:
            print top_feats[i]
    # considering samples names in the new table
    x = x[:, idx]
    var_names = (np.array(var_names)[idx]).tolist()

if TESTFILE is not None and TSLABELSFILE is not None:
    OUTFILE = os.path.join(OUTDIR, '_'.join([BASEFILE, "DNN", RANK_METHOD, SCALING]))
    mcc_cv_val_plot = open(OUTFILE+"_MCC_CV_VAL.txt", "w")
    x_axis = []
    y_axis = []
    point_names = []

    mcc_cv_val_plot.write("\t".join(mcc_cv_val[0])+"\n")
    mcc_cv_val = mcc_cv_val[1:]
    for line in mcc_cv_val:
        point_names.append(line[0])
        x_axis.append(line[1])
        y_axis.append(line[2])
        mcc_cv_val_plot.write("\t".join(line)+"\n")
    mcc_cv_val_plot.close()

'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.scatter(x_axis, y_axis, alpha=0.5)
    plt.title("MCC in CrossValidation VS MCC in Validation")
    plt.xlabel("MCC CrossValidation")
    plt.ylabel("MCC Validation")
    for label, x, y in zip(point_names, x_axis, y_axis):
        plt.annotate(
            label,
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    plt.savefig(OUTFILE+"_MCC_CV_VAL_PLOT.png")
'''

