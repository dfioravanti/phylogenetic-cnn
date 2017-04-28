## This code is written by Marco Chierici <chierici@fbk.eu>, Alessandro Zandona' <zandona@fbk.eu>.
## Based on code previously written by Davide Albanese.

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
import numpy as np
import csv
from sklearn_liblinear_tuning import svmlin_t
import os.path
from scaling import norm_l2
import mlpy
from input_output import load_data
import performance as perf
import sys
import argparse
from sklearn import svm
from sklearn import preprocessing
import ConfigParser
from distutils.version import StrictVersion
from collections import Counter
import tarfile
import glob

__version__ = '2.1'

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

parser = myArgumentParser(description='Run a training experiment (10x5-CV fold) using LibLinear.',
        fromfile_prefix_chars='@')
parser.add_argument('DATAFILE', type=str, help='Training datafile')
parser.add_argument('LABELSFILE', type=str, help='Sample labels')
parser.add_argument('OUTDIR', type=str, help='Output directory')
parser.add_argument('--solver', dest='SOLVER', type=str, choices=['l2r_l2loss_primal', 'l2r_l2loss_dual', 'l2r_l1loss_dual', 'l1r_l2loss_primal', 'l1r_lr_primal', 'l2r_lr_primal', 'l2r_lr_dual'], default='l2r_l2loss_dual', help='Solver type for linear SVM / Logistic Regression (default: %(default)s)')
parser.add_argument('--scaling', dest='SCALING', type=str, choices=['norm_l2', 'std', 'minmax', 'minmax0'], default='std', help='Scaling method (default: %(default)s)')
parser.add_argument('--ranking', dest='RANK_METHOD', type=str, choices=['SVM', 'ReliefF', 'tree', 'randomForest', 'KBest', 'random'], default='SVM', help='Feature ranking method: SVM (SVM weights), ReliefF, extraTrees, Random Forest, Anova F-score, random ranking (default: %(default)s)')
parser.add_argument('--random', action='store_true', help='Run with random sample labels')
parser.add_argument('--cv_k', type=np.int, default=5, help='Number of CV folds (default: %(default)s)')
parser.add_argument('--cv_n', type=np.int, default=10, help='Number of CV cycles (default: %(default)s)')
parser.add_argument('--reliefk', type=np.int, default=3, help='Number of nearest neighbors for ReliefF (default: %(default)s)')
parser.add_argument('--rfep', type=np.float, default=0.2, help='Fraction of features to remove at each iteration in RFE (p=0 one variable at each step, p=1 naive ranking) (default: %(default)s)')
parser.add_argument('--plot', action='store_true', help='Plot metric values over all training cycles' )
parser.add_argument('--quiet', action='store_true', help='Run quietly (no progress info)')

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
DATAFILE = args.DATAFILE
LABELSFILE = args.LABELSFILE
SOLVER = args.SOLVER
SCALING = args.SCALING
RANK_METHOD = args.RANK_METHOD
OUTDIR = args.OUTDIR
plot_out = args.plot
random_labels = args.random
CV_K = args.cv_k
CV_N = args.cv_n
relief_k = args.reliefk
rfe_p = args.rfep
QUIET = args.quiet

BASEFILE = os.path.splitext(os.path.basename(DATAFILE))[0]
OUTFILE = os.path.join(OUTDIR, '_'.join([BASEFILE, SOLVER, RANK_METHOD, SCALING]))

# create OUTDIR if not present
try:
    os.makedirs(OUTDIR)
except OSError:
    if not os.path.isdir(OUTDIR):
        raise

# parse the SOLVER string
PENALTY, LOSS, DUAL = SOLVER.split('_')
LogisticRegression = False
if LOSS == 'lr':
    LogisticRegression = True
    from sklearn import linear_model
if DUAL == 'dual':
    dual_flag = True
elif DUAL == 'primal':
    dual_flag = False

# load modules
if RANK_METHOD == 'ReliefF':
    from relief import ReliefF
    # add ReliefF K to OUTFILE
    OUTFILE = os.path.join(OUTDIR, '_'.join([BASEFILE, SOLVER, RANK_METHOD + str(relief_k), SCALING]))
elif RANK_METHOD == 'tree' :
    from sklearn.ensemble import ExtraTreesClassifier
elif RANK_METHOD == 'randomForest':
    from sklearn.ensemble import RandomForestClassifier
elif RANK_METHOD == 'KBest':
    from sklearn.feature_selection import SelectKBest, f_classif

# number of Montecarlo CV cycles (for SVM tuning)
TUN_CV_K = 10
# fraction of the dataset to keep apart as test split (for SVM tuning)
TUN_CV_P = 50
# list of C values for SVM tuning
TUN_SVM_C = [10**k for k in np.arange(-2, 3)]
#TUN_SVM_C = [10**k for k in np.arange(-7, 3)]

sample_names, var_names, x = load_data(DATAFILE)
y = np.loadtxt(LABELSFILE, dtype=np.int)

# build FSTEPS according to dataset size
nfeat = x.shape[1]
ord = np.int(np.log10(nfeat))
fs = np.empty(0, dtype=np.int)
for p in range(ord+1):
    fs = np.concatenate((fs, np.dot(10**p, np.arange(10))))
fs = np.unique(fs)[1:]
# cap FSTEPS at 10000 features, if applicable
FSTEPS = fs[ fs <= 10000 ].tolist() if nfeat>10000 else fs[ fs < nfeat ].tolist() + [nfeat]

# prepare output files
metricsf = open(OUTFILE + "_metrics.txt", 'w')
metrics_w = csv.writer(metricsf, delimiter='\t', lineterminator='\n')

rankingf = open(OUTFILE + "_featurelist.txt", 'w')
ranking_w = csv.writer(rankingf, delimiter='\t', lineterminator='\n')
ranking_w.writerow(["FEATURE_ID", "FEATURE_NAME", "MEAN_POS", "MEDIAN_ALL", "MEDIAN_0", "MEDIAN_1", "FOLD_CHANGE", "LOG2_FOLD_CHANGE"])

internalf = open(OUTFILE + "_internal.txt", 'w')
internal_w = csv.writer(internalf, delimiter='\t', lineterminator='\n')
internal_w.writerow(["TUN_CV_K: %d, TUN_CV_P: %d" % (TUN_CV_K, TUN_CV_P)])
internal_w.writerow(["ITERATION", "FOLD", "C", "MCC (test)", "ERR (test)", "MCC (train)", "ERR (train)"])

stabilityf = open(OUTFILE + "_stability.txt", 'w')
stability_w = csv.writer(stabilityf, delimiter='\t', lineterminator='\n')

# prepare output arrays
RANKING = np.empty((CV_K*CV_N, x.shape[1]), dtype=np.int)
NPV = np.empty((CV_K*CV_N, len(FSTEPS)))
PPV = np.empty_like(NPV)
SENS = np.empty_like(NPV)
SPEC = np.empty_like(NPV)
MCC = np.empty_like(NPV)
AUC = np.empty_like(NPV)
DOR = np.empty_like(NPV)
ACC = np.empty_like(NPV)

ys=[]

if random_labels:
    #np.random.seed(1234)
    tmp = y.copy()
    np.random.shuffle(tmp)
    for i in range(CV_N):
        ys.append(tmp)
else:
    for i in range(CV_N):
        ys.append(y)

for n in range(CV_N):
    idx = mlpy.cv_kfold(n=x.shape[0], k=CV_K, strat=ys[n], seed=n)
    print "=" * 80
    print "%d over %d experiments" % (n+1, CV_N)

    for i, (idx_tr, idx_ts) in enumerate(idx):

        if not QUIET:
            print "_" * 80
            print "-- %d over %d folds" % (i+1, CV_K)

        x_tr, x_ts = x[idx_tr], x[idx_ts]
        y_tr, y_ts = ys[n][idx_tr], ys[n][idx_ts]

        if not QUIET:
            print "Tuning SVM..."

        C, mcc, err, mcc_tr, err_tr = svmlin_t(x_tr, y_tr, LOSS=LOSS[:2], PENALTY=PENALTY[:2], DUAL=dual_flag, scaling=SCALING, list_C=TUN_SVM_C, cv_k=TUN_CV_K, cv_p=TUN_CV_P, logReg=LogisticRegression, quiet=QUIET)

        if not QUIET:
            print "Best C: %s (MCC: %.3f)" % (C, mcc)
        internal_w.writerow([n+1, i+1, C, mcc, err, mcc_tr, err_tr])

        # centering and normalization
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


        if LogisticRegression:
            clf = linear_model.LogisticRegression(C=C, penalty=PENALTY[:2], dual=dual_flag, class_weight='auto', tol=1e-2)
        else:
            clf = svm.LinearSVC(C=C, loss=LOSS[:2], penalty=PENALTY[:2], dual=dual_flag, class_weight='auto', tol=1e-2)
        
        if RANK_METHOD == 'random':
            ranking_tmp = np.arange(len(var_names))
            np.random.seed((n*CV_K)+i)
            np.random.shuffle(ranking_tmp)
        elif RANK_METHOD == 'SVM':
            clf.fit(x_tr, y_tr)
            weights = (clf.coef_).reshape(-1)
            ranking_tmp = np.argsort(np.abs(weights))[::-1]
        elif RANK_METHOD == 'ReliefF':
            relief = ReliefF(relief_k, seed=n)
            relief.learn(x_tr, y_tr)
            w = relief.w()
            ranking_tmp = np.argsort(w)[::-1]
        elif RANK_METHOD == 'tree' :
            forest = ExtraTreesClassifier(n_estimators=250, criterion='gini', random_state=n)
            forest.fit(x_tr, y_tr)
            ranking_tmp = np.argsort(forest.feature_importances_)[::-1]
        elif RANK_METHOD == 'randomForest' :
            forest = RandomForestClassifier(n_estimators=250, criterion='gini', random_state=n)
            forest.fit(x_tr, y_tr)
            ranking_tmp = np.argsort(forest.feature_importances_)[::-1]
        elif RANK_METHOD == 'KBest':
            selector = SelectKBest(f_classif)
            selector.fit(x_tr, y_tr)
            ranking_tmp = np.argsort( -np.log10(selector.pvalues_) )[::-1]
            
        RANKING[(n * CV_K) + i] = ranking_tmp

        for j, s in enumerate(FSTEPS):
            v = RANKING[(n * CV_K) + i][:s]
            x_tr_fs, x_ts_fs = x_tr[:, v], x_ts[:, v]
            clf.fit(x_tr_fs, y_tr)
            p = clf.predict(x_ts_fs)
            pv = -clf.decision_function(x_ts_fs).reshape(-1,1)
            NPV[(n * CV_K) + i, j] = perf.npv(y_ts, p)
            PPV[(n * CV_K) + i, j] = perf.ppv(y_ts, p)
            SENS[(n * CV_K) + i, j] = perf.sensitivity(y_ts, p)
            SPEC[(n * CV_K) + i, j] = perf.specificity(y_ts, p)
            MCC[(n * CV_K) + i, j] = perf.KCCC_discrete(y_ts, p)
            AUC[(n * CV_K) + i, j] = perf.auc_wmw(y_ts, -pv)
            DOR[(n * CV_K) + i, j] = perf.dor(y_ts, p)
            ACC[(n * CV_K) + i, j] = perf.accuracy(y_ts, p)

internalf.close()

# write metrics for all CV iterations
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
NPVCI = []
for i in range(NPV.shape[1]):
    NPVCI.append(mlpy.bootstrap_ci(NPV[:, i]))
PPVCI = []
for i in range(PPV.shape[1]):
    PPVCI.append(mlpy.bootstrap_ci(PPV[:, i]))
SENSCI = []
for i in range(SENS.shape[1]):
    SENSCI.append(mlpy.bootstrap_ci(SENS[:, i]))
SPECCI = []
for i in range(SPEC.shape[1]):
    SPECCI.append(mlpy.bootstrap_ci(SPEC[:, i]))
MCCCI = []
for i in range(MCC.shape[1]):
    MCCCI.append(mlpy.bootstrap_ci(MCC[:, i]))
AUCCI = []
for i in range(AUC.shape[1]):
    AUCCI.append(mlpy.bootstrap_ci(AUC[:, i]))
DORCI = []
for i in range(DOR.shape[1]):
    DORCI.append(mlpy.bootstrap_ci(DOR[:, i]))
ACCCI = []
for i in range(ACC.shape[1]):
    ACCCI.append(mlpy.bootstrap_ci(ACC[:, i]))

# Borda list
BORDA_ID, _, BORDA_POS = mlpy.borda_count(RANKING)
# optimal number of features (yielding max MCC)
opt_feats = FSTEPS[np.argmax(AMCC)]
# optimal C (occurring most times in the CV)
intMCC = np.loadtxt(OUTFILE + "_internal.txt", delimiter='\t', usecols=(2,), skiprows=2)
cntmcc = Counter( intMCC )
opt_C = cntmcc.most_common()[0][0]

# Canberra stability indicator
STABILITY = []
PR = np.argsort( RANKING )
for ss in FSTEPS:
    STABILITY.append( mlpy.canberra_stability(PR, ss) )

metrics_w.writerow(["FS_WITH_BEST_MCC", opt_feats])
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

for j, s in enumerate(FSTEPS):
    metrics_w.writerow([s,
                        AMCC[j], MCCCI[j][0], MCCCI[j][1],
                        ASENS[j], SENSCI[j][0], SENSCI[j][1],
                        ASPEC[j], SPECCI[j][0], SPECCI[j][1],
                        APPV[j], PPVCI[j][0], PPVCI[j][1],
                        ANPV[j], NPVCI[j][0], NPVCI[j][1],
                        AAUC[j], AUCCI[j][0], AUCCI[j][1],
                        AACC[j], ACCCI[j][0], ACCCI[j][1],
                        ADOR[j], DORCI[j][0], DORCI[j][1],
                        ADOR_APPROX[j] ])
    stability_w.writerow( [s, STABILITY[j]] )

metricsf.close()
stabilityf.close()

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
config.set("CV PARAMETERS", "TUN_SVM_C", TUN_SVM_C)
config.add_section("INPUT")
config.set("INPUT", "Data", os.path.realpath( DATAFILE ))
config.set("INPUT", "Labels", os.path.realpath( LABELSFILE ))
config.set("INPUT", "Solver_type", SOLVER)
config.set("INPUT", "Scaling", SCALING)
config.set("INPUT", "Rank_method", RANK_METHOD)
config.set("INPUT", "Random_labels", random_labels)
config.add_section("OUTPUT")
config.set("OUTPUT", "Metrics", os.path.realpath( OUTFILE + "_metrics.txt" ))
config.set("OUTPUT", "Borda", os.path.realpath( OUTFILE + "_featurelist.txt" ))
config.set("OUTPUT", "Internal", os.path.realpath( OUTFILE + "_internal.txt" ))
config.set("OUTPUT", "Stability", os.path.realpath( OUTFILE + "_stability.txt" ))
config.set("OUTPUT", "MCC", np.max(AMCC))
config.set("OUTPUT", "N_feats", opt_feats)
config.set("OUTPUT", "opt_C", opt_C)
config.write(logf)
logf.close()


if plot_out:
    from metrics_plot import *
    plt_title = (' ').join( [os.path.basename(DATAFILE).replace('.txt', ''), SCALING, SOLVER] )
    if random_labels:
        metplot(RLFile = (OUTFILE + "_metrics.txt"), title = plt_title)
    elif RANK_METHOD=='random':
        metplot(RRFile = (OUTFILE + "_metrics.txt"), title = plt_title)
    else: 
        metplot(normFile = (OUTFILE + "_metrics.txt"), title = plt_title)
