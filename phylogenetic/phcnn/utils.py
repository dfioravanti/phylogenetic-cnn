
import csv
import numpy as np


def load_datafile(filename, dtype=np.float):

    """
    Load a specific file which we assume have the following structure:
        the first row contains the names of the features,
        the first column contains the names for the samples,
        the remaining entries are the data with type dtype.
    """

    with open(filename, 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter='\t')

        feature_names = next(reader)[1:]
        sample_names, data = [], []

        for row in reader:
            sample_names.append(row[0])
            data.append([float(elem) for elem in row[1:]])

    return feature_names, sample_names, np.array(data, dtype=dtype)
