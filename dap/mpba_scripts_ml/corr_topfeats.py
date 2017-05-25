## Compute Pearson correlation between top ranked features and all the other ones.

import numpy as np
import argparse
import sys 
import ConfigParser
import os
from extract_topfeats_nowr import extract_feats
from scipy.stats import pearsonr
import csv

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


parser = argparse.ArgumentParser(description='Compute Pearson correlation between top feats and the other ones.')
parser.add_argument('--abundTable', dest='ABUND_TABLE', type=str, help='Table with absolute reads abundance (samples X features). If not given as input, training dataset will be considered')
parser.add_argument('CONFIGFILE', type=str, help='Training experiment configuration file')
parser.add_argument('OUTFOLDER', type=str, help='Output folder')
parser.add_argument('--corr_thr', dest='CORR_THR', type=np.float, default=0.7, help='Correlation threshold (default: %(default)s)')
parser.add_argument('--nfeat', dest='NFEATS', type=np.int, default=15, help='Number of top features considered (default: %(default)s)')

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
abundTable = args.ABUND_TABLE
outFld = args.OUTFOLDER
CONFIGFILE = args.CONFIGFILE
corr_thr = args.CORR_THR
NFEATS = args.NFEATS

config = ConfigParser.RawConfigParser()
config.read(CONFIGFILE)

if not config.has_section('INPUT'):
    print "%s is not a valid configuration file." % CONFIGFILE
    sys.exit(3)

# if matrix with ABSOLUTE reads abundance is not given, consider the normalized matrix specified in log file
if abundTable is not None:
    ab_matrix = abundTable
else:
    ab_matrix = config.get("INPUT", "data")

RANK = config.get("OUTPUT", "Borda")
NFEATS = config.getint("OUTPUT", "N_feats")

# consider the top ranked features
TRFILE = ab_matrix
top_feats = extract_feats(TRFILE, RANK, NFEATS)

# consider the other features
rankList = np.loadtxt(RANK, delimiter='\t', skiprows=1, dtype=str)

ALL_FEATS = rankList.shape[0]
all_feats = extract_feats(TRFILE, RANK, ALL_FEATS)
notop_feats = all_feats[:,NFEATS+1:]

#outFile = os.path.join(outPath, os.path.basename(ab_matrix).replace('.txt', '_corr.txt'))
outFile = os.path.join(outFld, os.path.basename(ab_matrix).replace('.txt', '_topFeatsCorrel.txt'))

outw = open(outFile, 'w')
writer = csv.writer(outw, delimiter='\t', lineterminator='\n')

# header of matrix with correlation between top feats and the other ones
writer.writerow(['FEATURES'] + notop_feats[0,:].tolist())

correl = []
pcc_over_thr = []
# NON top ranked features highly correlated (PCC > corr_thr with) with top ranked features
feats_over_thr = []
# top ranked features highly correlated (PCC > corr_thr with) with non top ranked features 
topfeats_overcorr = []
# position of the highly correlated NON top features
position_ntf = []
# position of the highly correlated TOP features
position_tf = []
# compute Pearson correlation 
for tf in range(1, top_feats.shape[1]):
    x = top_feats[1:, tf]
    for ntf in range(0, notop_feats.shape[1]):
        y = notop_feats[1:, ntf]
        pcoeff = pearsonr(x.astype(np.float), y.astype(np.float))[0]
        correl.append(pcoeff)
        if (abs(pcoeff)> corr_thr):
            pcc_over_thr.append(pcoeff)
            position_tf.append(tf)
            feats_over_thr.append(notop_feats[0, ntf])
            topfeats_overcorr.append(top_feats[0, tf])
            position_ntf.append(NFEATS+1+ntf)
    writer.writerow([ top_feats[0,tf] ] + correl)
    correl = []

outw.close()


# file with features with PCC > corr_thr with top ranked features
outFile_over_thr = os.path.join(outFld, os.path.basename(ab_matrix).replace('.txt', '_overcorr_' + str(corr_thr) +'.txt'))
outw_thr = open(outFile_over_thr, 'w') 
writer_thr = csv.writer(outw_thr, delimiter='\t', lineterminator='\n')
# header
writer_thr.writerow(['Rank of top feature', 'Top feature', 'Rank of non top feature', 'Non top feature', 'Pearson Correlation Coefficient'])

for j in range(0, len(pcc_over_thr)):
    writer_thr.writerow([position_tf[j], topfeats_overcorr[j] , position_ntf[j], feats_over_thr[j], pcc_over_thr[j]])

outw_thr.close()
