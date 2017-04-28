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

from __future__ import division
import numpy as np
import os.path
import mlpy
from input_output import load_data
import performance as perf
from scaling import *
import argparse
import sys
import ConfigParser
from sklearn.ensemble import RandomForestClassifier
from distutils.version import StrictVersion
from extract_topfeats import extract_feats

parser = argparse.ArgumentParser(description='Run a validation experiment using LibLinear.')
parser.add_argument('CONFIGFILE', type=str, help='Training experiment configuration file')
parser.add_argument('TSFILE', type=str, help='Validation datafile')
parser.add_argument('OUTDIR', type=str, help='Output directory')
parser.add_argument('--tslab', type=str, default=None, help='Validation labels, if available')
parser.add_argument('--nf', type=int, default=None, help='Custom number of top features')

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
CONFIGFILE = vars(args)['CONFIGFILE']
TSFILE = vars(args)['TSFILE']
OUTDIR = vars(args)['OUTDIR']
TSLABELSFILE = vars(args)['tslab']
NFEATS = vars(args)['nf']

config = ConfigParser.RawConfigParser()
config.read(CONFIGFILE)
if not config.has_section('INPUT'):
    print "%s is not a valid configuration file." % CONFIGFILE
    sys.exit(3)

TRFILE = config.get("INPUT", "Data")
LABELSFILE = config.get("INPUT", "Labels")
RANK = config.get("OUTPUT", "Borda")
if NFEATS is None:
    NFEATS = config.getint("OUTPUT", "N_feats")

BASEFILE = os.path.splitext(TRFILE)[0]
OUTFILE = os.path.join(OUTDIR, os.path.basename(BASEFILE))

# extract the top-ranked NFEATS features from TRAINING set
TR_TOPFEATS = OUTFILE + '_top%s_tr.txt' % NFEATS 
extract_feats(TRFILE, RANK, NFEATS, TR_TOPFEATS)
# extract the top-ranked NFEATS features from VALIDATION set
TS_TOPFEATS = OUTFILE + '_top%s_ts.txt' % NFEATS
extract_feats(TSFILE, RANK, NFEATS, TS_TOPFEATS)

# load data
sample_names_tr, var_names_tr, x_tr = load_data(TR_TOPFEATS)
y_tr = np.loadtxt(LABELSFILE, dtype=np.int, delimiter='\t')
sample_names_ts, var_names_ts, x_ts = load_data(TS_TOPFEATS)
# load the TS labels if available
if TSLABELSFILE is not None:
    y_ts = np.loadtxt(TSLABELSFILE, dtype=np.int, delimiter='\t')

# prediction
forest = RandomForestClassifier(n_estimators=500, criterion='gini', random_state=0)
forest.fit(x_tr, y_tr)

p_tr = forest.predict(x_tr)
p_ts = forest.predict(x_ts)

prob_tr = forest.predict_proba(x_tr)
prob_ts = forest.predict_proba(x_ts)

print "MCC on train: %.3f" % (perf.KCCC_discrete(y_tr, p_tr))
if TSLABELSFILE is not None:
    print "MCC on validation: %.3f" % (perf.KCCC_discrete(y_ts, p_ts))

# write output files
fout = open(OUTFILE + "_TEST_pred_tr.txt", "w")
for i in range(len(sample_names_tr)):
    fout.write("%s\t%i\n" % (sample_names_tr[i], p_tr[i]))
fout.close()

fout = open(OUTFILE + "_TEST_pred_ts.txt", "w")
for i in range(len(sample_names_ts)):
    fout.write("%s\t%i\n" % (sample_names_ts[i], p_ts[i]))
fout.close()

np.savetxt(OUTFILE + "_TEST_signature.txt",
           np.array(var_names_tr).reshape(-1,1),
           fmt='%s', delimiter='\t')

fout = open(OUTFILE + "_TEST_prob_tr.txt", "w")
fout.write("SAMPLE\tCLASS 0\tCLASS 1\n")
for i in range(len(sample_names_tr)):
    fout.write("%s\t%f\t%f\n" % (sample_names_tr[i], prob_tr[i,0], prob_tr[i,1]))
fout.close()

fout = open(OUTFILE + "_TEST_prob_ts.txt", "w")
fout.write("SAMPLE\tCLASS 0\tCLASS 1\n")
for i in range(len(sample_names_ts)):
    fout.write("%s\t%f\t%f\n" % (sample_names_ts[i], prob_ts[i,0], prob_ts[i,1]))
fout.close()

