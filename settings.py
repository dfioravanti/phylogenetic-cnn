import os
from utils import merge_file

# ============================================
# -- Directory & Paths Section
# ============================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ============================================
# -- DATA Section
# ============================================

DATABASE_NAME = 'camda'

TRAINING_PREFIX = 'TR'
TEST_PREFIX = 'TS'

TRAINING_COORDINATES_FILEPATH = os.path.join(DATA_DIR, 'coordinates', DATABASE_NAME,
                                    ''.join(['MDSmat', TRAINING_PREFIX, '.txt']))

TEST_COORDINATES_FILEPATH = os.path.join(DATA_DIR, 'coordinates', DATABASE_NAME,
                                    ''.join(['MDSmat', TEST_PREFIX, '.txt']))

TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, 'expressions', DATABASE_NAME, 
				      ''.join(['dataExpr', TRAINING_PREFIX, '.txt']))
TRAINING_LABELS_FILEPATH = os.path.join(DATA_DIR, 'expressions', DATABASE_NAME, 
                                        ''.join(['dataExpr', TRAINING_PREFIX, '_labels.txt']))


TEST_DATA_FILEPATH = os.path.join(DATA_DIR, 'expressions', DATABASE_NAME, 
                                  ''.join(['dataExpr', TEST_PREFIX, '.txt']))
TEST_LABELS_FILEPATH = os.path.join(DATA_DIR, 'expressions', DATABASE_NAME, 
                                    ''.join(['dataExpr', TEST_PREFIX, '_labels.txt']))

# =======================================
# Multi-Layer Perceptron Specific Section
# =======================================

mlp_hidden_units=128

# ==================================
# -- PhyloCNN Model Specific Section
# ==================================

# No. of Convolutional Filters to use.
# This should be a list of lists, a list is a phylo-convolutional block followed by a dropout
# for example [8, 16, 32] will generate a
# phyloconv1D with 8 filters -> phyloconv1D with 16 filters -> phyloconv1D with 32 filters -> dropout
# any new block will attached to the previous one.
nb_convolutional_filters = [16, 16]

# No. of Neighbours
# The structures should be the same as nb_convolutional_filters any number represents the number of
# neighbours for that phyloconv1D
nb_phylo_neighbours = [4, 4]

# ==================================
# -- SVM Model Specific Section
# ==================================

# scorer
# --------------------------
from sklearn.metrics import make_scorer, matthews_corrcoef
scorer = make_scorer(matthews_corrcoef)

# parameters
tuning_parameter = [{'SVM__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

# Tuning cross validation
# --------------------------
# No. Split
tuning_cv_K = 5
# Percent test split
tuning_cv_P = 0.1

# kernel
# --------------------------
kernel = 'rbf'
