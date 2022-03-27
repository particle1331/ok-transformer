# inefficient-networks

This repo contains a collection of Jupyter notebooks that run end-to-end after some minimal initial setup (e.g. setting up the directory structure and downloading the required datasets). The name of this collection is inspired by the following study:


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

Note that [`requirements.txt`](https://github.com/particle1331/inefficient-networks/blob/dev/package/requirements/requirements.txt) has TensorFlow commented out &mdash; this only serves to specify the version used in the notebooks. Installing TensorFlow can be different depending on the operating system and available hardware, so I leave this task to the user.