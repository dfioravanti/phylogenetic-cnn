# Install MLPY 3.5.0 Dependency 

`mlpy` package is required for some operations included in the DAP procedure.

To install `mlpy` dependency, the following steps must be accomplished:

## Install from Source

The `mlpy` package available on PyPI is outdated and not working on OSX platforms.

Therefore, we _recommend_ to install `mlpy` directly from the source code, using the `mlpy-3.5.0.tar.gz` archive 
available in this repo.

In more details, these are the steps to follow:

	```
        tar xvzf mlpy-3.5.0.tar.gz
        cd mlpy-3.5.0
        python setup.py install
        ```

## Test the Installation

To verify that `mlpy` package has been properly installed, type the following command in a terminal 

	```
        python -c "import mlpy; print(mlpy.__version__);"
	```

