#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from input_output import get_data
from phcnn.dap import dap
from phcnn import settings


def _creation_output_dir(output_dir):

    if not settings.overwrite and os.path.isdir(output_dir):
        answer = input('Output directory already existing, overwrite all files? [Y/n] ')
        if not(answer == 'Y' or answer == 'y' or answer == ''):
            sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)


def main():

    _creation_output_dir(settings.OUTPUT_DIR)
    datafile = settings.TRAINING_DATA_FILEPATH
    labels_datafile = settings.TRAINING_LABELS_FILEPATH
    coordinates_datafile = settings.COORDINATES_FILEPATH
    test_datafile = settings.TEST_DATA_FILEPATH
    test_label_datafile = settings.TEST_LABELS_FILEPATH

    inputs = get_data(datafile, labels_datafile, coordinates_datafile,
                      test_datafile, test_label_datafile)

    dap_model = dap(inputs)

if __name__ == '__main__':
    main()

