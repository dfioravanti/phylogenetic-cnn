#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from phylogenetic.input_output import get_data, create_parser
from phylogenetic.phcnn.globalsettings import GlobalSettings
import phylogenetic.phcnn.phcnn as phcnn

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

    inputs = get_data(GlobalSettings.datafile,
                    GlobalSettings.label_datafile,
                    GlobalSettings.coordinates_datafile)

    print(inputs)

    model = phcnn.PhcnnBuilder.build()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(inputs['xs'][1:20], inputs['ys'][1:20],
              batch_size=2,
              nb_epoch=2,
              validation_data=(inputs['xs'][21:40], inputs['ys'][21:40])
              )


if __name__ == '__main__':
    main()

