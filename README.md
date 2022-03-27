# inefficient-networks

This repo contains a collection of Jupyter notebooks that run end-to-end after some minimal initial setup (e.g. setting up the directory structure and downloading the required datasets). The name of this collection is inspired by the following study:


```
Brackbill D, Centola D (2020) Impact of network structure on collective 
learning: An experimental study in a data science competition. PLoS ONE 
15(9): e0237978. https://doi.org/10.1371/journal.pone.0237978
```


## Installation

The package [`inefficient_networks`](https://github.com/particle1331/inefficient-networks/tree/dev/package) sets up the directory structure and provides functions for downloading data from Kaggle, as well as other utility functions. 
To install:

```
$ git clone git@github.com:particle1331/inefficient-networks.git
$ cd inefficient-networks
$ cd package
$ pip install -e .
```

Note that the [`requirements.txt`](https://github.com/particle1331/inefficient-networks/blob/dev/package/requirements/requirements.txt) file has TensorFlow commented out. The installation can be different for each operating system and available hardware, so I leave installation of TensorFlow to the reader.
