#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from phylogenetic.input_output import get_data, create_parser
from phylogenetic.phcnn.globalsettings import GlobalSettings


def data():
    """
    Data providing function required by hyperas:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    return get_data(GlobalSettings.datafile,
                    GlobalSettings.labels_datafile,
                    GlobalSettings.coordinates_datafile)


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

    data = get_data(GlobalSettings.datafile,
                    GlobalSettings.label_datafile,
                    GlobalSettings.coordinates_datafile)

    print(data)

if __name__ == '__main__':
    main()

