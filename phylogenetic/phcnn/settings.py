

import os

# ============================================
# -- Directory & Paths Section
# ============================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')

# ============================================
# -- DAP Section
# ============================================

# -- Feature Scaling Method
# Choices are: std, minmax0, norm_l2, minmax
feature_scaling_method = 'std'

# -- Feature Ranking Method
# Choices are: ReliefF, random, KBest
feature_ranking_method = 'ReliefF'

# Only used with ReliefF
relief_k = 3  # K parameter for the (internal) KNN

# Only used with KBest
kbest = 10

# Percentage of total features to consider in the ranking list
feature_selection_percentage = [5, 25, 50, 75, 100]
# Decide whether to include the first Top feature resulting from ranking
include_top_feature = False

# random labels
use_random_labels = False  # Set to True to randomly shuffle training set labels

# -- Cross Validation
# No. Repetitions
Cv_N = 2
# No. Fold
Cv_K = 2
# Whether to apply Stratification
stratified = True

# =================================================
# -- Extra Settings (for pretty-printing & logging)
# =================================================

quiet = False  # verbosity
overwrite = True  # if True, the output_dir will be always created/overwritten regardless if it already exists

# ============================================
# -- Machine Learning Models Section
# ============================================

# ML Method
# Choices are: phcnn (Phylogenetic CNN), rf (random forest), svm (Support Vector Machine)
ml_model = 'phcnn'

# -- phcnn settings
epochs = 2
batch_size = 32
optimizer = 'adam'  # Choices are: sgd, adam, rmsprop
nb_phyloconv_layers = 2
verbose = 2 if quiet else 1

# No. of Convolutional Filters to use.
# Note: This could be either a number (scalar) or a list. If scalar, the **same** number will
# be used for all the PhyloConv Layers
nb_convolutional_filters = 2

# No. of Neighbours
# Note: This could be either a number (scalar) or a list. If scalar, the **same** number will
# be used for all the PhyloConv Layers
nb_phylo_neighbours = 2  # No. Neighbours

# -- rf settings

# -- svm settings

# ============================================
# -- DATA Section
# ============================================

DISEASE = 'HS_CDf'

TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, 'true_data', DISEASE,
                                      'Sokol_16S_taxa_HS_CDflare_commsamp_training.txt')
COORDINATES_FILEPATH = os.path.join(DATA_DIR, 'coordinates', 'coordinates_cdf.txt')
TRAINING_LABELS_FILEPATH = os.path.join(DATA_DIR, 'true_data', DISEASE,
                                        'Sokol_16S_taxa_HS_CDflare_commsamp_training_lab.txt')

TEST_DATA_FILEPATH = os.path.join(DATA_DIR, 'true_data', DISEASE,
                                  'Sokol_16S_taxa_HS_CDflare_commsamp_validation.txt')
TEST_LABELS_FILEPATH = os.path.join(DATA_DIR, 'true_data', DISEASE,
                                    'Sokol_16S_taxa_HS_CDflare_commsamp_validation_lab.txt')