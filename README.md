# Phylogenetic Convolutional Neural Network

A novel architecture for metagenomic classification defined as phylogenetic convolutional neural network, as presented in the paper 
[Phylogenetic Convolutional -Neural Networks in Metagenomics](https://arxiv.org/abs/1709.02268).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

## Clone this repo

    ```
    git clone https://gitlab.fbk.eu/MPBA/phylogenetic-cnn.git
    ``` 
    
The DAP (Data Analysis Protocol) Project is included in this repo as an external reference (i.e. *Git Submodule*).

Therefore, the first time this repo is cloned, the Git Submodule must be initialised - after the `clone` command, 
you should see a `dap` directory in your cloned copy which is empty. 

Thus:

* `cd dap`
* `git submodule init`
* `git submodule update`

#### Alternatively:

You could do the same operations in just one line:

`git clone --recursive https://gitlab.fbk.eu/MPBA/phylogenetic-cnn.git`

### Prerequisites

A complete conda environment is provided as a `.yml` file in the folder `envs`. 

Additionally it is required to install the `mlpy` library. Further instructions to install **MLPY 3.5.0** Python package are reported in the 
[`README.md`](envs/deps/README.md) file, in the `envs/deps` folder.

## Replication Package
| disease |                                                                           |                                                                              |                                                                                 |                                                                                    |
|:-------:|---------------------------------------------------------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
|   cdf   | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACZ292WW9Qd0oteFE) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGUXhwbmpJQ0VMNVU) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACM3F1M0dIM3NkX0k) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGckJjWlBYQ2gyYjQ) |
|   cdr   | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACLTJfaEYwNlNZaDg) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGbVdVNHprSFpvdTg) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACZHNvd3pocG94RXM) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGU2RHN2VXbWhsdk0) |
|   icdf  | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACa0I1XzNCSnIzNDA) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGSjhhOVJFQUZUMHc) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACTjhBck5fVDV3QnM) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGT2VndEp2QzdSZHM) |
|   icdr  | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACa0dsRjVzTzkydlk) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGWjlqdmhucnRFRUU) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACc2JScTRqUXJfWjQ) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGZ1ZhWVpDQ2oza1E) |
|   ucf   | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACYWVEOHZzVWZzM1k) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGYUc2cng5OUpKTk0) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACOW5rNjAxX1pQbUE) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGX1RNNWhWM1RsREU) |
|   ucr   | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACRFpwQTVYa1h1ekE) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGZEJ0eTVEem1QR28) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACc25OclVvald0N28) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGaWFVbUg0Wm9WNFU) |

## Running Experiments

### Runners

One can select the algorithm (SVM, random forrest, MLP, ph-cnn) to be used by simply decide which runner to execute. 

* `multilayerperceptron_runner.py`: Multi-Layer Perceptron
* `phylocnn_runner.py`: Phylogenetic Convolutional Neural Network
* `randomforest_runner.py`: Random forest
* `svm_runner.py`: Support Vector Machine
* `transfer_learning_runner.py`: Phylogenetic Convolutional Neural Network used for transfer learning. It assumes that pre-trained network weights are provided
(see `weights` folder).

### Settings

In order to configure how the program runs one needs to modify the following files:

* `settings.py` - where it can be chose which type of data we want to load, where are the data, where to output, etc...
* `dap/settings.py` - where it can be set how the DAP is supposed to operate. More informations are available in the readme in the `dap` folder and in the paper.
* `dap/deep_learning_settings.py` - where all the settings specific for deep learning can be set.

## Notebooks

* PhyloConv1D: In this notebook we report code examples and explanations on how to use the new PhyloConv1D Keras layer.
We use experimental data, as examples.

* Embedding_ICDF: In this notebook we report results and plots of embeddings of Phylo-Convolutional Layers 
calculated on data of ICDf disease included in the IBD dataset, as reported in the paper.

## Authors

* Valerio Maggio - [website](http://github.com/leriomaggio)
* Diego Fioravanti- [website](https://is.tuebingen.mpg.de/people/dfioravanti)

## License

This project is licensed under _GNU General Public License v3.0_ [GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/#) - see the [LICENSE.txt](LICENSE.txt) file for details
