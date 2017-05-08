#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from phcnn.input_output import create_parser
from phcnn.globalsettings import GlobalSettings
import phcnn.phcnn as phcnn

from phylogenetic.phcnn.globalsettings import GlobalSettings
from phylogenetic.phcnn.input_output import get_data


def main():

    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    inputs = parser.parse_args()
    GlobalSettings.set(inputs)
    # os.makedirs(GlobalSettings.output_directory, exist_ok=True)

    print(GlobalSettings.settings_to_strings())
    print(sys.argv)

    data = phcnn.get_data(GlobalSettings.datafile,
                          GlobalSettings.label_datafile,
                          GlobalSettings.coordinates_datafile)

    print(data)

if __name__ == '__main__':
    main()


def data():
    """
    Data providing function required by hyperas:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    return get_data(GlobalSettings.datafile,
                    GlobalSettings.labels_datafile,
                    GlobalSettings.coordinates_datafile)