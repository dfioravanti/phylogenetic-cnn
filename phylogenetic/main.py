#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

from input_output import get_data, create_parser
from phcnn.globalsettings import GlobalSettings
from phcnn.dap import DAP
from keras.utils import np_utils
from keras.optimizers import SGD, Adam


def main2():

    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    settings = parser.parse_args()
    GlobalSettings.set(settings)
    # os.makedirs(GlobalSettings.output_directory, exist_ok=True)

    inputs = get_data(GlobalSettings.datafile,
                      GlobalSettings.label_datafile,
                      GlobalSettings.coordinates_datafile,
                      GlobalSettings.validation_datafile,
                      GlobalSettings.validations_labels_datafile)

    model = phcnn.PhcnnBuilder.build(
                                     nb_coordinates=inputs['nb_coordinates'],
                                     nb_features=inputs['nb_features'],
                                     nb_outputs=2
                                     )

    #opt = SGD(lr=0.001, nesterov=True, momentum=0.8, decay=1e-06)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    nb_training = 60
    xs_tr = inputs['xs'][0:nb_training]
    ys_tr = np_utils.to_categorical(inputs['ys'][0:nb_training])
    xs_ts = inputs['xs'][nb_training:]
    ys_ts = np_utils.to_categorical(inputs['ys'][nb_training:])

    model.fit({'xs_input': xs_tr,
               'coordinates_input': inputs['coordinates'][0:nb_training]},
              {'output': ys_tr},
              epochs=200,
              verbose=2,
              validation_data=({'xs_input': inputs['validation_xs'],
                                'coordinates_input': inputs['coordinates'][0:inputs['validation_xs'].shape[0]]},
                               {'output': np_utils.to_categorical(inputs['validation_ys'])}
                              )
              )

    score, acc = model.evaluate({'xs_input': xs_ts,
                                 'coordinates_input': inputs['coordinates'][nb_training:]},
                                 {'output': ys_ts}, verbose=0)
    print('Test accuracy:', acc)


def _creation_output_dir(output_dir):
    if os.path.isdir(output_dir):
        answer = input('Output directory already existing, overwrite all files? [Y/n] ')
        if not(answer == 'Y' or answer == 'y' or answer == ''):
            sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)


def main():
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    settings = parser.parse_args()
    GlobalSettings.set(settings)

    _creation_output_dir(GlobalSettings.output_directory)

    inputs = get_data(GlobalSettings.datafile,
                      GlobalSettings.label_datafile,
                      GlobalSettings.coordinates_datafile,
                      GlobalSettings.validation_datafile,
                      GlobalSettings.validations_labels_datafile)

    DAP(inputs)


if __name__ == '__main__':
    main()

