#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from input_output import get_data, create_parser, get_configuration
from phcnn.globalsettings import GlobalSettings
from phcnn.dap import dap


def _creation_output_dir(output_dir):

    if not GlobalSettings.overwrite and os.path.isdir(output_dir):
        answer = input('Output directory already existing, overwrite all files? [Y/n] ')
        if not(answer == 'Y' or answer == 'y' or answer == ''):
            sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)


def main():

    GlobalSettings.set(get_configuration())

    _creation_output_dir(GlobalSettings.output_directory)

    inputs = get_data(GlobalSettings.datafile,
                      GlobalSettings.label_datafile,
                      GlobalSettings.coordinates_datafile,
                      GlobalSettings.validation_datafile,
                      GlobalSettings.validations_labels_datafile)

    dap(inputs)


if __name__ == '__main__':
    main()

