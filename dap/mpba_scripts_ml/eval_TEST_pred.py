import argparse
import sys
import pandas as pd
from sklearn.metrics import matthews_corrcoef

parser = argparse.ArgumentParser(description='Evaluate predictions using true labels.')
parser.add_argument('PREDFILE', type=str, help='Predicted labels (two-column tab-separated file: sample name, label)')
parser.add_argument('LABELSFILE', type=str, help='True sample labels (two-column tab-separated file: sample name, label)')

if len(sys.argv)==1:
    parser.print_help()
    sys.exit(1)

args = parser.parse_args()
test_pred_f = args.PREDFILE
true_labels_f = args.LABELSFILE

df = pd.read_csv(true_labels_f, delimiter='\t', header=None, names=['class'], index_col=0)
dfp = pd.read_csv(test_pred_f, delimiter='\t', header=None, names=['class_pred'], index_col=0)

# join dataframes
joined = pd.concat([df, dfp], join='inner', axis=1)
# evaluate performance
MCC = matthews_corrcoef( joined['class'], joined['class_pred'] )

print "MCC = %.5f" % MCC
