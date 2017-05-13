#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from phylogenetic.input_output import get_data, create_parser
from phylogenetic.phcnn.globalsettings import GlobalSettings
import phylogenetic.phcnn.phcnn as phcnn
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras import backend as K



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
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    model.fit({'xs_input': inputs['xs'],
               'coordinates_input': inputs['coordinates']},
              {'output': np_utils.to_categorical(inputs['ys'])},
              epochs=200,
              validation_data=({'xs_input': inputs['validation_xs'],
                                'coordinates_input': inputs['coordinates'][0:inputs['validation_xs'].shape[0]]},
                               {'output': np_utils.to_categorical(inputs['validation_ys'])}
                              )
              )

if __name__ == '__main__':
    main()

