# 𝗜𝗻𝗲𝗳𝗳𝗶𝗰𝗶𝗲𝗻𝘁 𝗡𝗲𝘁𝘄𝗼𝗿𝗸𝘀


[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fparticle1331%2Finefficient-networks%2Fbadge%3Fref%3Dmaster&label=build&logo=none)](https://actions-badge.atrox.dev/particle1331/inefficient-networks/goto?ref=master)
[![Stars](https://img.shields.io/github/stars/particle1331/inefficient-networks?style=social)](https://github.com/particle1331/inefficient-networks)



This is a collection of notebooks on **machine learning engineering**. 
Each notebook should run end-to-end after installing the [`inefficient_networks`](https://github.com/particle1331/inefficient-networks/tree/master/src/inefficient_networks) package. This package provides utilities for downloading datasets from Kaggle and reducing boilerplate code in setting up the coding environment. Datasets used are from [Kaggle](https://www.kaggle.com/datasets) or from an open-source library such as
[`torchvision`](https://pytorch.org/vision/stable/index.html) and [`tensorflow_datasets`](https://www.tensorflow.org/datasets). 


The name of this collection is inspired by the following study:


```txt
Brackbill D, Centola D (2020) Impact of network structure on collective 
learning: An experimental study in a data science competition. PLoS ONE 
15(9): e0237978. https://doi.org/10.1371/journal.pone.0237978
```

<br>

```{figure} img/pone.0237978.g003.png
---
width: 30em
name: study
---
**Evolution of solution discovery for members of the two groups.** The efficient network converged on a small set of solutions, whereas individuals in the inefficient network explored a greater diversity of solutions, and eventually converged on the best solution. [Fig. 3 in the study]
```

## Installation

Installing the package:

```
$ git clone git@github.com:particle1331/inefficient-networks.git
$ cd inefficient-networks
$ cd src
$ pip install -e .
```

Building the book:

```
$ # cd to the root directory
$ make docs
```



## Hardware

```text
Model Name:	                MacBook Air
Model Identifier:	        MacBookAir10,1
Chip:                           Apple M1
Total Number of Cores:          8 (4 performance and 4 efficiency)
Memory:                         8 GB
System Firmware Version:	7429.61.2
OS Loader Version:	        7429.61.2
```




## References 

```{bibliography}
```