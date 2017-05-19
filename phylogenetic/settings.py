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


# ==================================
# -- PhyloCNN Model Specific Section
# ==================================

# No. of Convolutional Filters to use.
# Note: This could be either a number (scalar) or a list. If scalar, the **same** number will
# be used for all the PhyloConv Layers
nb_convolutional_filters = [2]

# No. of Neighbours
# Note: This could be either a number (scalar) or a list. If scalar, the **same** number will
# be used for all the PhyloConv Layers
nb_phylo_neighbours = [2]  # No. Neighbours