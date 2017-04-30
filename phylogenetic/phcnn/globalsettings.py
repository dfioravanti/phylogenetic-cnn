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

    @staticmethod
    def set(inputs):
        """
        As input this function requires a Namespace object produced by the ArgumentParser from the argparse library.
        """

        GlobalSettings.datafile = inputs.DATAFILE
        GlobalSettings.coordinates_datafile = inputs.COORDINATES
        GlobalSettings.label_datafile = inputs.LABELSFILE
        GlobalSettings.scaling = inputs.SCALING
        GlobalSettings.rank_method = inputs.RANK_METHOD
        GlobalSettings.output_directory = inputs.OUTDIR
        GlobalSettings.is_output_plotted = inputs.plot
        GlobalSettings.cv_n = inputs.cv_n
        GlobalSettings.cv_k = inputs.cv_k
        GlobalSettings.relief_k = inputs.reliefk
        GlobalSettings.rfe_p = inputs.rfep
        GlobalSettings.is_quiet = inputs.quiet
        GlobalSettings.validation_datafile = inputs.tsfile
        GlobalSettings.validations_labels_datafile = inputs.tslab
        GlobalSettings.trials = inputs.trials

    @staticmethod
    def settings_to_strings():
        return 'Datafile : {}' \
               '\nCoordinate datafile : {}' \
               '\nLabel datafile : {}' \
               '\nOutput directory : {}' \
               '\nIs the output plotted? : {}' \
               '\nScaling : {}' \
               '\nRank method : {}' \
               '\nCV_N : {}' \
               '\nCV_K : {}' \
               '\nNearest neighbors ReliefF : {}' \
               '\nPercentage dropped per iteration of RFE : {}' \
               '\nIs quiet ?: {}' \
               '\nValidation datafile : {}' \
               '\nValidation labels datafile : {}' \
               '\nHypersearch trials : {}'.format(
                            GlobalSettings.datafile,
                            GlobalSettings.coordinates_datafile,
                            GlobalSettings.label_datafile,
                            GlobalSettings.output_directory,
                            GlobalSettings.is_output_plotted,
                            GlobalSettings.scaling,
                            GlobalSettings.rank_method,
                            GlobalSettings.cv_n,
                            GlobalSettings.cv_k,
                            GlobalSettings.relief_k,
                            GlobalSettings.rfe_p,
                            GlobalSettings.is_quiet,
                            GlobalSettings.validation_datafile,
                            GlobalSettings.validations_labels_datafile,
                            GlobalSettings.trials
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
