## This code is written by Davide Albanese <albanese@fbk.eu>
import numpy as np
import csv

def load_data(filename):
    f = open(filename, 'r')
    csv_r = csv.reader(f, delimiter='\t')
    var_names = csv_r.next()[1:]
    sample_names, data = [], []
    for row in csv_r:
        sample_names.append(row[0])
        data.append([float(elem) for elem in row[1:]])
    return sample_names, var_names, np.array(data)

