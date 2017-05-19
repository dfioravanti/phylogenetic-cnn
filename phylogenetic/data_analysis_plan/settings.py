# ============================================
# -- DAP Section
# ============================================

# -- Feature Scaling Method

# Control whether to apply Feature Scaling
# Default: True
apply_feature_scaling = True

# Choices are: std, minmax0, norm_l2, minmax
# -- Feature Scaling Choices
STD = 'std'
MINMAX0 = 'minmax0'
MINMAX = 'minmax'
NORM_L2 = 'norm_l2'

feature_scaling_method = STD

# -- Feature Ranking Method

# Choices are: ReliefF, random, KBest
RELIEFF = 'ReliefF'
RANDOM = 'rnd'
KBEST = 'KBest'
feature_ranking_method = RELIEFF

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

# Categorical Labels: Decide whether to one-hot-encode labels
to_categorical = True

# -- Cross Validation
# No. Repetitions
Cv_N = 1
# No. Fold
Cv_K = 2

# Whether to apply Stratification
stratified = True

# =================================================
# -- Machine Learning Models Hyperparameter Section
# =================================================

# -- Optimizers Choices
ADAM = 'adam'
SGD = 'sgd'
RMSPROP = 'rmsprop'

# -- Optimisers Specific Settings

# == SGD ==
nesterov = True
sgd_lr = 0.001
momentum = 0.9
decay = 1e-06

# == RMSPROP ==
rmsprop_epsilon = 1e-08
rmsprop_lr = 0.001
rho = 0.9

# == ADAM ==
adam_epsilon = 1e-08
adam_lr = 0.001
beta_1 = 0.9
beta_2 = 0.999

# Fit Settings
epochs = 2
batch_size = 32
optimizer = 'adam'  # Choices are: sgd, adam, rmsprop
fit_verbose = 2

