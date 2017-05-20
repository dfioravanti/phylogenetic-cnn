# ============================================
# A. DAP Section
# ============================================

# ---------------------------
# 0. Cross Validation Section
# ---------------------------

# No. Repetitions
Cv_N = 1
# No. Fold
Cv_K = 2

# Apply Stratification to labels when
# generating folds.
stratified = True

# Enable/Disable Use of Random Labels
# If True, a random shuffling to training set labels
# will be applied.
use_random_labels = False

# Categorical Labels:
# Decide whether to apply one-hot-encode to labels
to_categorical = True

# --------------------------
# 1. Feature Scaling Section
# --------------------------

# Enable/Disable Feature Scaling
apply_feature_scaling = True

# Feature Scaling method
# ----------------------
# This can either be a string or
# an actual sklearn Transformer object
# (see sklearn.preprocessing)
from .scaling import StandardScaler
feature_scaler = StandardScaler(copy=False)

# --------------------------
# 2. Feature Ranking Section
# --------------------------

# This can be eitehr a string or a
# (custom) function object,
from .ranking import kbest_ranking
feature_ranker = kbest_ranking

# -----------------
# 2.1 Feature Steps
# _________________

# Ranges (expressed as percentage wrt. the total)
# of features to consider when generating feature steps

# Default: 5%, 25%, 50%, 75%, 100% (all)
feature_ranges = [5, 25, 50, 75, 100]

# Include top feature in the feature steps
use_top_feature = False

# =================================================
# B. Machine Learning Models HyperParameter Section
# =================================================

# Use of scikit Pipelines!!

# =================================================
# C. Deep Learning Models Hyperparameter Section
# =================================================

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# This following section contains settings
# that will be used only by the
# `DeepLearningDAP` class!
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-==-=-=

# -------------------
# 1 Model Fit Section
# ___________________

# No. of Epochs
epochs = 2

# Size of the Batch
batch_size = 32

# Verbosity level of `fit` method
fit_verbose = 2

# Validation split
# ----------------
# (Automatically ignored if `validation_data` is provided)
validation_split = 0.0  # Default: no split

# Shuffle (boolean)
# -----------------
# Whether to shuffle the samples at each epoch.
shuffle = True  # Shuffle samples at each epoch, by default.

# Class Weights
# -------------
# Dictionary mapping classes to a weight value
# used for scaling the loss function (during training only).
class_weight = None

# Sample Weights
# --------------
# Numpy array of weights for the training samples,
# used for scaling the loss function (during training only).
# You can either pass a flat (1D) Numpy array with the same
# length as the input samples
# (1:1 mapping between weights and samples),
# or in the case of temporal data, you can pass a 2D array
# with shape (samples, sequence_length), to
# apply a different weight to every timestep of every sample.
# In this case you should make sure to specify
# sample_weight_mode="temporal" in compile().
sample_weight = None

# Initial Epoch
# -------------
# Epoch at which to start training
# (useful for resuming a previous training run).
initial_epoch = 0  # Default: 0 -  first epoch!

# Additional Callbacks
# --------------------
# By default, the `keras.callbacks.ModelSelection` callback
# will be applied at each fit, in addition to the default
# `keras.callbacks.History`.
#
# To automatically plug additional callbacks into
# model fit, please add configured keras Callbacks objects
# in the list below
callbacks = []

# -----------------------
# 2 Model Compile Section
# _______________________

# Loss Function
# --------------
# This may be either a string or a function object
# (see keras.losses for examples)
loss='categorical_crossentropy'

# Loss Weights
# -------------
# List of weights to associate to losses
# (in case of multi-output networks)
loss_weights = None

# Additional Compile Settings
# ---------------------------
# Additional Compile settings directly
# passed to Theano functions. Ignored by Tensorflow
extra_compilation_parameters = {}

# Loss Metric(s)
# --------------
# List of metrics to optimise in the training.
# These can either be strings or function objects
# Default metric is accuracy.
# (see keras.metrics for examples)
metrics = ['accuracy']

# Optimizer
# ---------
# This can either be a string or an
# optimizer object (see keras.optimizers)
from keras.optimizers import Adam
optimizer = Adam(lr=0.001, decay=1e-06,
                 epsilon=1e-08, beta_1=0.9, beta_2 = 0.999)