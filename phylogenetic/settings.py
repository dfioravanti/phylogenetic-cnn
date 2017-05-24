import os

# ============================================
# -- Directory & Paths Section
# ============================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ============================================
# -- DATA Section
# ============================================

DISEASE = 'CDf'

# two type of data avaiable: true_data or synthetic_data
TYPE_DATA = 'synthetic_data'

# This can be empty, if so the datasets will be located in HD_DISEASE/
# instead of HD_DISEASE/NB_SAMPLES
NB_SAMPLES = '200'

TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, TYPE_DATA, ''.join(['HS_', DISEASE]), NB_SAMPLES,
                                      ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training.txt']))
COORDINATES_FILEPATH = os.path.join(DATA_DIR, 'coordinates',
                                    ''.join(['coordinates_', DISEASE.lower(), '.txt']))
TRAINING_LABELS_FILEPATH = os.path.join(DATA_DIR, TYPE_DATA, ''.join(['HS_', DISEASE]), NB_SAMPLES,
                                        ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training_lab.txt']))

TEST_DATA_FILEPATH = os.path.join(DATA_DIR, TYPE_DATA, ''.join(['HS_', DISEASE]), NB_SAMPLES,
                                  ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test.txt']))
TEST_LABELS_FILEPATH = os.path.join(DATA_DIR, TYPE_DATA, ''.join(['HS_',DISEASE]), NB_SAMPLES,
                                    ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test_lab.txt']))

# ==================================
# -- PhyloCNN Model Specific Section
# ==================================

# No. of Convolutional Filters to use.
# Note: This could be either a number (scalar) or a list. If scalar, the **same** number will
# be used for all the PhyloConv Layers
nb_convolutional_filters = [8, 8, 16]

# No. of Neighbours
# Note: This could be either a number (scalar) or a list. If scalar, the **same** number will
# be used for all the PhyloConv Layers
nb_phylo_neighbours = [4, 4, 4]  # No. Neighbours