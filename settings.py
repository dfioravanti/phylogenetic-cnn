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

DISEASE = 'CDf'

# two type of data avaiable: true_data or synthetic_data

TRUE_DATA = 'true_data'
SYNT_DATA = 'synthetic_data'
MIX_DATA = 'mix_data'
# Choose one of the two above!
TYPE_DATA = MIX_DATA

# This can be empty, if so the datasets will be located in HD_DISEASE/
# instead of HD_DISEASE/NB_SAMPLES
NB_SAMPLES = '200'

COORDINATES_FILEPATH = os.path.join(DATA_DIR, 'coordinates',
                                    ''.join(['coordinates_', DISEASE.lower(), '.txt']))

if TYPE_DATA == TRUE_DATA or TYPE_DATA == SYNT_DATA:

    if TYPE_DATA == SYNT_DATA:
        DISEASE_FOLDER = os.path.join(''.join(['HS_', DISEASE]), NB_SAMPLES)
    else:
        DISEASE_FOLDER = ''.join(['HS_', DISEASE])

    TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, TYPE_DATA, DISEASE_FOLDER,
                                          ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training.txt']))
    TRAINING_LABELS_FILEPATH = os.path.join(DATA_DIR, TYPE_DATA, DISEASE_FOLDER,
                                            ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training_lab.txt']))

    TEST_DATA_FILEPATH = os.path.join(DATA_DIR, TYPE_DATA, DISEASE_FOLDER,
                                      ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test.txt']))
    TEST_LABELS_FILEPATH = os.path.join(DATA_DIR, TYPE_DATA, DISEASE_FOLDER,
                                        ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test_lab.txt']))

elif TYPE_DATA == MIX_DATA:

    SYNT_DISEASE_FOLDER = os.path.join(''.join(['HS_', DISEASE]), NB_SAMPLES)
    MIX_DISEASE_FOLDER = os.path.join(DATA_DIR, TYPE_DATA, SYNT_DISEASE_FOLDER)
    TRUE_DISEASE_FOLDER = ''.join(['HS_', DISEASE])

    os.makedirs(MIX_DISEASE_FOLDER, exist_ok=True)

    SYNT_TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, SYNT_DATA, SYNT_DISEASE_FOLDER,
                                               ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training.txt']))
    SYNT_TEST_DATA_FILEPATH = os.path.join(DATA_DIR, SYNT_DATA, SYNT_DISEASE_FOLDER,
                                           ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test.txt']))
    TRAINING_DATA_FILEPATH = os.path.join(MIX_DISEASE_FOLDER, 'training.txt')
    merge_file(SYNT_TRAINING_DATA_FILEPATH, SYNT_TEST_DATA_FILEPATH,
               TRAINING_DATA_FILEPATH, skip_header_second_file=True)

    SYNT_TRAINING_LABELS_FILEPATH = os.path.join(DATA_DIR, SYNT_DATA, SYNT_DISEASE_FOLDER,
                                                 ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training_lab.txt']))
    SYNT_TEST_LABELS_FILEPATH = os.path.join(DATA_DIR, SYNT_DATA, SYNT_DISEASE_FOLDER,
                                             ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test_lab.txt']))
    TRAINING_LABELS_FILEPATH = os.path.join(MIX_DISEASE_FOLDER, 'training_lab.txt')
    merge_file(SYNT_TRAINING_LABELS_FILEPATH, SYNT_TEST_LABELS_FILEPATH,
               TRAINING_LABELS_FILEPATH)

    TRUE_TRAINING_DATA_FILEPATH = os.path.join(DATA_DIR, TRUE_DATA, TRUE_DISEASE_FOLDER,
                                               ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training.txt']))
    TRUE_TEST_DATA_FILEPATH = os.path.join(DATA_DIR, TRUE_DATA, TRUE_DISEASE_FOLDER,
                                           ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test.txt']))
    TEST_DATA_FILEPATH = os.path.join(MIX_DISEASE_FOLDER, 'test.txt')
    merge_file(TRUE_TRAINING_DATA_FILEPATH, TRUE_TEST_DATA_FILEPATH,
               TEST_DATA_FILEPATH, skip_header_second_file=True)

    TRUE_TRAINING_LABELS_FILEPATH = os.path.join(DATA_DIR, TRUE_DATA, TRUE_DISEASE_FOLDER,
                                                 ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_training_lab.txt']))
    TRUE_TEST_LABELS_FILEPATH = os.path.join(DATA_DIR, TRUE_DATA, TRUE_DISEASE_FOLDER,
                                             ''.join(['Sokol_16S_taxa_HS_', DISEASE, '_commsamp_test_lab.txt']))
    TEST_LABELS_FILEPATH = os.path.join(MIX_DISEASE_FOLDER, 'test_label.txt')
    merge_file(TRUE_TRAINING_LABELS_FILEPATH, TRUE_TEST_LABELS_FILEPATH,
               TEST_LABELS_FILEPATH)
else:
    raise Exception("Select a correct type of data")

# ==================================
# -- PhyloCNN Model Specific Section
# ==================================

# No. of Convolutional Filters to use.
# This should be a list of lists, a list is a phylo-convolutional block followed by a dropout
# for example [8, 16, 32] will generate a
# phyloconv1D with 8 filters -> phyloconv1D with 16 filters -> phyloconv1D with 32 filters -> dropout
# any new block will attached to the previous one.
nb_convolutional_filters = [32, 64]

# No. of Neighbours
# The structures should be the same as nb_convolutional_filters any number represents the number of
# neighbours for that phyloconv1D
nb_phylo_neighbours = [4, 4]
