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
import os.path
from scaling import norm_l2
import mlpy
from input_output import load_data
import performance as perf
import sys
import argparse
import ConfigParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

__version__ = '2.0'

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

parser = myArgumentParser(description='Run a training experiment (10x5-CV fold) using Random Forest as classifier.',
        fromfile_prefix_chars='@')
parser.add_argument('DATAFILE', type=str, help='Training datafile')
parser.add_argument('LABELSFILE', type=str, help='Sample labels')
parser.add_argument('OUTDIR', type=str, help='Output directory')
parser.add_argument('--scaling', dest='SCALING', type=str, choices=['norm_l2', 'std', 'minmax'], default='std', help='Scaling method (default: %(default)s)')
parser.add_argument('--ranking', dest='RANK_METHOD', type=str, choices=['ReliefF', 'tree', 'randomForest', 'KBest', 'random'], default='randomForest', help='Feature ranking method: ReliefF, extraTrees, Random Forest, Anova F-score, random ranking (default: %(default)s)')
parser.add_argument('--random', action='store_true', help='Run with random sample labels')
parser.add_argument('--cv_k', type=np.int, default=5, help='Number of CV folds (default: %(default)s)')
parser.add_argument('--cv_n', type=np.int, default=10, help='Number of CV cycles (default: %(default)s)')
parser.add_argument('--reliefk', type=np.int, default=3, help='Number of nearest neighbors for ReliefF (default: %(default)s)')
parser.add_argument('--plot', action='store_true', help='Plot metric values over all training cycles' )

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
random_labels = args.random
CV_K = args.cv_k
CV_N = args.cv_n
relief_k = args.reliefk

BASEFILE = os.path.splitext(os.path.basename(DATAFILE))[0]
SVM_TYPE = 'RandomForest'
OUTFILE = os.path.join(OUTDIR, '_'.join([BASEFILE, SVM_TYPE, 'RF' if RANK_METHOD=='randomForest' else RANK_METHOD, SCALING]))

# load modules
if RANK_METHOD == 'ReliefF':
    from relief import ReliefF
    # add ReliefF K to OUTFILE
    OUTFILE = os.path.join(OUTDIR, '_'.join([BASEFILE, SVM_TYPE, RANK_METHOD + str(relief_k), SCALING]))
elif RANK_METHOD == 'tree' :
    from sklearn.ensemble import ExtraTreesClassifier
elif RANK_METHOD == 'KBest':
    from sklearn.feature_selection import SelectKBest, f_classif

# number of Montecarlo CV cycles (for SVM tuning)
TUN_CV_K = 10
# fraction of the dataset to keep apart as test split (for SVM tuning)
TUN_CV_P = 50

sample_names, var_names, x = load_data(DATAFILE)
y = np.loadtxt(LABELSFILE, dtype=np.int)

# build FSTEPS according to dataset size
nfeat = x.shape[1]
ord = np.int(np.log10(nfeat))
fs = np.empty(0, dtype=np.int)
for p in range(ord+1):
    fs = np.concatenate( (fs, np.dot(10**p, np.arange(10))) )
fs = np.unique(fs)[1:]
# cap FSTEPS at 10000 features, if applicable
FSTEPS = fs[ fs <= 10000 ].tolist() if nfeat>10000 else fs[ fs < nfeat ].tolist() + [nfeat]

# prepare output files
metricsf = open(OUTFILE + "_metrics.txt", 'w')
metrics_w = csv.writer(metricsf, delimiter='\t', lineterminator='\n')

rankingf = open(OUTFILE + "_featurelist.txt", 'w')
ranking_w = csv.writer(rankingf, delimiter='\t', lineterminator='\n')
ranking_w.writerow(["FEATURE_ID", "FEATURE_NAME", "MEAN_POS", "MEDIAN_ALL", "MEDIAN_0", "MEDIAN_1", "FOLD_CHANGE", "LOG2_FOLD_CHANGE"])

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
    np.random.seed(0)
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

        print "_" * 80
        print "-- %d over %d folds" % (i+1, CV_K)

        x_tr, x_ts = x[idx_tr], x[idx_ts]
        y_tr, y_ts = ys[n][idx_tr], ys[n][idx_ts]

        forest = RandomForestClassifier(n_estimators=500, criterion='gini', random_state=n, n_jobs=2)
        forest.fit(x_tr, y_tr)
        
        if RANK_METHOD == 'random':
            ranking_tmp = np.arange(len(var_names))
            np.random.seed((n*CV_K)+i)
            np.random.shuffle(ranking_tmp)
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
            ranking_tmp = np.argsort(forest.feature_importances_)[::-1]
        elif RANK_METHOD == 'KBest':
            selector = SelectKBest(f_classif)
            selector.fit(x_tr, y_tr)
            ranking_tmp = np.argsort( -np.log10(selector.pvalues_) )[::-1]
            
        RANKING[(n * CV_K) + i] = ranking_tmp

        #forest = RandomForestClassifier(n_estimators=500, criterion='gini')
        for j, s in enumerate(FSTEPS):
            v = RANKING[(n * CV_K) + i][:s]
            x_tr_fs, x_ts_fs = x_tr[:, v], x_ts[:, v]
            forest.fit(x_tr_fs, y_tr)
            p = forest.predict(x_ts_fs)
            NPV[(n * CV_K) + i, j] = perf.npv(y_ts, p)
            PPV[(n * CV_K) + i, j] = perf.ppv(y_ts, p)
            SENS[(n * CV_K) + i, j] = perf.sensitivity(y_ts, p)
            SPEC[(n * CV_K) + i, j] = perf.specificity(y_ts, p)
            MCC[(n * CV_K) + i, j] = perf.KCCC_discrete(y_ts, p)
            AUC[(n * CV_K) + i, j] = roc_auc_score(y_ts, p)
            DOR[(n * CV_K) + i, j] = perf.dor(y_ts, p)
            ACC[(n * CV_K) + i, j] = perf.accuracy(y_ts, p)

# write metrics for all CV iterations
np.savetxt(OUTFILE + "_allmetrics_MCC.txt", MCC, fmt='%.4f', delimiter='\t')
np.savetxt(OUTFILE + "_allmetrics_SENS.txt", SENS, fmt='%.4f', delimiter='\t')
np.savetxt(OUTFILE + "_allmetrics_SPEC.txt", SPEC, fmt='%.4f', delimiter='\t')
np.savetxt(OUTFILE + "_allmetrics_PPV.txt", PPV, fmt='%.4f', delimiter='\t')
np.savetxt(OUTFILE + "_allmetrics_NPV.txt", NPV, fmt='%.4f', delimiter='\t')
np.savetxt(OUTFILE + "_allmetrics_AUC.txt", AUC, fmt='%.4f', delimiter='\t')
np.savetxt(OUTFILE + "_allmetrics_ACC.txt", ACC, fmt='%.4f', delimiter='\t')
np.savetxt(OUTFILE + "_allmetrics_DOR.txt", DOR, fmt='%.4f', delimiter='\t')

# write all rankings
np.savetxt(OUTFILE + "_ranking.csv", RANKING, fmt='%d', delimiter='\t')

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
config.add_section("INPUT")
config.set("INPUT", "Data", os.path.realpath( DATAFILE ))
config.set("INPUT", "Labels", os.path.realpath( LABELSFILE ))
config.set("INPUT", "Classifier", "RandomForest")
config.set("INPUT", "n_estimators", 500)
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
config.write(logf)
logf.close()


if plot_out:
    from metrics_plot import *
    plt_title = (' ').join( [os.path.basename(DATAFILE).replace('.txt', ''), SCALING, SVM_TYPE] )
    if random_labels:
        metplot(RLFile = (OUTFILE + "_metrics.txt"), title = plt_title)
    elif RANK_METHOD=='random':
        metplot(RRFile = (OUTFILE + "_metrics.txt"), title = plt_title)
    else: 
        metplot(normFile = (OUTFILE + "_metrics.txt"), title = plt_title)
