#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from phylogenetic.input_output import get_data, create_parser
from phylogenetic.phcnn.globalsettings import GlobalSettings
import phylogenetic.phcnn.phcnn as phcnn
from keras.utils import np_utils
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
                      GlobalSettings.coordinates_datafile)

    model = phcnn.PhcnnBuilder.build(
                                     nb_coordinates=inputs['nb_coordinates'],
                                     nb_features=inputs['nb_features'],
                                     nb_outputs=2
                                     )

    intermediate_output = model.predict({'xs_input': inputs['xs'],
                                         'coordinates_input': inputs['coordinates']})

    print(inputs['xs'].shape)
    print(intermediate_output[0,0,:,0].shape)
    print(intermediate_output[:,0,:,0])


    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # print(model.summary())
    #
    # model.fit({'xs_input': inputs['xs']},
    #           {'output': np_utils.to_categorical(inputs['ys'])},
    #           batch_size=inputs['nb_samples'],
    #           epochs=10
    #           #validation_data=(inputs['xs'][21:40], inputs['ys'][21:40])
    #           )


if __name__ == '__main__':
    main()

