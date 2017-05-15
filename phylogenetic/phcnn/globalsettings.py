class GlobalSettings:

    """

    This object will contain the settings as parsed from command line. We need such global object since we cannot pass
    parameters to the data() function required by hyperas.

    """

    datafile = None
    coordinates_datafile = None
    label_datafile = None
    scaling = None
    rank_method = None
    output_directory = None
    is_output_plotted = None
    cv_n = None
    cv_k = None
    relief_k = None
    rfe_p = None
    is_quiet = None
    validation_datafile = None
    validations_labels_datafile = None
    trials = None
    overwrite = None

    @staticmethod
    def set(inputs):
        """
        As input this function requires a Namespace object produced by the ArgumentParser from the argparse library.
        """

        GlobalSettings.datafile = inputs['data']
        GlobalSettings.coordinates_datafile = inputs['coordinates']
        GlobalSettings.label_datafile = inputs['labels']
        GlobalSettings.output_directory = inputs['output_dir']
        GlobalSettings.validation_datafile = inputs['validation_data']
        GlobalSettings.validations_labels_datafile = inputs['validation_labels']

        GlobalSettings.scaling = inputs['scaling']
        GlobalSettings.rank_method = inputs['rank_method']
        GlobalSettings.cv_n = int(inputs['cv_n'])
        GlobalSettings.cv_k = int(inputs['cv_k'])
        GlobalSettings.relief_k = int(inputs['reliefk'])

        if inputs['quiet'] == 'True':
            GlobalSettings.is_quiet = True
        else:
            GlobalSettings.is_quiet = False

        if inputs['overwrite'] == 'True':
            GlobalSettings.overwrite = True
        else:
            GlobalSettings.overwrite = False

    @staticmethod
    def settings_to_strings():
        return 'Datafile : {}' \
               '\nCoordinate datafile : {}' \
               '\nLabel datafile : {}' \
               '\nOutput directory : {}' \
               '\nWill overwrite output?: {}'\
               '\nScaling : {}' \
               '\nRank method : {}' \
               '\nCV_N : {}' \
               '\nCV_K : {}' \
               '\nNearest neighbors ReliefF : {}' \
               '\nIs quiet ?: {}' \
               '\nValidation datafile : {}' \
               '\nValidation labels datafile : {}'.format(
                            GlobalSettings.datafile,
                            GlobalSettings.coordinates_datafile,
                            GlobalSettings.label_datafile,
                            GlobalSettings.output_directory,
                            GlobalSettings.overwrite,
                            GlobalSettings.scaling,
                            GlobalSettings.rank_method,
                            GlobalSettings.cv_n,
                            GlobalSettings.cv_k,
                            GlobalSettings.relief_k,
                            GlobalSettings.is_quiet,
                            GlobalSettings.validation_datafile,
                            GlobalSettings.validations_labels_datafile,
                            )


if __name__ == '__main__':

    import parser
    import sys

    parser = parser.create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    inputs = parser.parse_args()

    GlobalSettings.set(inputs)
    print(GlobalSettings.settings_to_strings())
    print(sys.argv)
