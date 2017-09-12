# Phylogenetic Convolutional Neural Network

A novel architecture for metagenomic classification defined as phylogenetic convolutional neural network, as presented in the paper [Phylogenetic Convolutiona -Neural Networks in Metagenomics](https://arxiv.org/abs/1709.02268).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Cloning 

We recommend to clone the repository with the --recursive option in order to clone the submodule too.

### Prerequisites

A complete conda environment is provided as a yml file in the folder `envs`. Additionally it is required to install the mlpy library as described in the readme included in the `envs/deps` folder.

## Running the program

In order to run the program one needs to modify the following files:

* `settings.py` - where it can be chose which type of data we want to load, where are the data, where to output, etc...
* `dap/settings.py` - where it can be set how the DAP is supposed to operate. More informations are available in the readme in the `dap` folder and in the paper.
* `dap/deep_learning_settings.py` - where all the settings specific for deep learning can be set.

Once done that one can select the algormthm (SVM, random forrest, MLP, ph-cnn) to be used by simply decide which runner to execute. For example:

```
python3 phylocnn_runner.py
```
will execute the phylogenetic convolutional neural network on the selected data with the selected settings.

## Replication
| disease |                                                                           |                                                                              |                                                                                 |                                                                                    |
|:-------:|---------------------------------------------------------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
|   cdf   | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACZ292WW9Qd0oteFE) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGUXhwbmpJQ0VMNVU) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACM3F1M0dIM3NkX0k) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGckJjWlBYQ2gyYjQ) |
|   cdr   | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACLTJfaEYwNlNZaDg) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGbVdVNHprSFpvdTg) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACZHNvd3pocG94RXM) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGU2RHN2VXbWhsdk0) |
|   icdf  | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACa0I1XzNCSnIzNDA) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGSjhhOVJFQUZUMHc) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACTjhBck5fVDV3QnM) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGT2VndEp2QzdSZHM) |
|   icdr  | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACa0dsRjVzTzkydlk) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGWjlqdmhucnRFRUU) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACc2JScTRqUXJfWjQ) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGZ1ZhWVpDQ2oza1E) |
|   ucf   | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACYWVEOHZzVWZzM1k) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGYUc2cng5OUpKTk0) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACOW5rNjAxX1pQbUE) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGX1RNNWhWM1RsREU) |
|   ucr   | [IBD data](https://drive.google.com/open?id=0B5ihbogwrsACRFpwQTVYa1h1ekE) | [IBD results](https://drive.google.com/open?id=0BwWtRh6l0dHGZEJ0eTVEem1QR28) | [Synthetic data](https://drive.google.com/open?id=0B5ihbogwrsACc25OclVvald0N28) | [Synthetic results](https://drive.google.com/open?id=0BwWtRh6l0dHGaWFVbUg0Wm9WNFU) |

## Notebooks

The folder `notebook` contains some notebook that will make it clearer how the program works.

* PhyloConv1D: Explanation of how to use the PhyloConv1D layer
* Embedding_ICDF: TODO

## Authors

* Valerio Maggio - [website]()
* Diego Fioravanti- [website](https://is.tuebingen.mpg.de/people/dfioravanti)

## License

This project is licensed under SOMETHING - see the [LICENSE.md](LICENSE.md) file for details
