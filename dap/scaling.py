"""Module containing all supported
feature scaling methods.

This module defines all the objects and 
functions to be used in the settings 
related to "feature scaling".
"""
from sklearn.preprocessing import (Normalizer, StandardScaler, MinMaxScaler)


# Aliases - to support string values in settings!
std = STD = StandardScaler
minmax = MINMAX = MinMaxScaler
norm = NORM = Normalizer


_REVERSE_NAMES_MAP = {
    hash(std.__name__): 'std',
    hash(minmax.__name__): 'minmax',
    hash(norm.__name__): 'norm'
}


def get_feature_scaling_name(reference):
    """This function is basically identical
    to `get_feature_ranking_name` function 
    defined in `rankings`.
    It returns a textual representation of 
    selected feature scaling method, provided
    in input as reference.

    Parameters
    ----------
    reference: object (i.e. function | class)
        Reference to the feature scaling method 
        selected in settings.

    Returns
    -------
    str:
        Textual name of scaling method
    """

    return _REVERSE_NAMES_MAP.get(hash(type(reference).__name__), 'no_name')
