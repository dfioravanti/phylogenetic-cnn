#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from phcnn.parser import create_parser
from phcnn.globalsettings import GlobalSettings
import phcnn.phcnn as phcnn


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
