# inefficient-networks

This is a collection of notebooks on machine learning theory and engineering. 
Each notebook should run end-to-end after installing the [`inefficient_networks`](https://github.com/particle1331/inefficient-networks/tree/dev/package) utilities library (see below).
Datasets come from [Kaggle](https://www.kaggle.com/) or as part of some open-source library such as
[`torchvision`](https://pytorch.org/vision/stable/index.html) and [`tensorflow_datasets`](https://www.tensorflow.org/datasets). The name of this collection is inspired by the following study:


```
Brackbill D, Centola D (2020) Impact of network structure on collective 
learning: An experimental study in a data science competition. PLoS ONE 
15(9): e0237978. https://doi.org/10.1371/journal.pone.0237978
```


## Installation

The package [`inefficient_networks`](https://github.com/particle1331/inefficient-networks/tree/dev/package) provides functions for downloading data from Kaggle and a [`config`](https://github.com/particle1331/inefficient-networks/blob/dev/package/inefficient_networks/config.py) object to reduce boilerplate code in setting up the coding environment. 
To install:

```
$ git clone git@github.com:particle1331/inefficient-networks.git
$ cd inefficient-networks
$ cd package
$ pip install -e .
```

Note that TensorFlow has been commented out in [`requirements.txt`](https://github.com/particle1331/inefficient-networks/blob/dev/package/requirements/requirements.txt) &mdash; this only serves to specify the version used in the notebooks. Installing TensorFlow depends on the specific operating system and available 
hardware, so I leave this task to the reader.

## Hardware

The notebooks are tested to run on an M1 Macbook Air. Colab and Kaggle kernels have similar capacity, so the notebooks should be able to run in these environments without crashing.

```
Model Name:               MacBook Air
Model Identifier:         MacBookAir10,1
Chip:                     Apple M1
Total Number of Cores:    8 (4 performance and 4 efficiency)
Memory:                   8 GB
System Firmware Version:  7429.61.2
OS Loader Version:        7429.61.2
```
