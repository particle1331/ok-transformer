# 𝗜𝗻𝗲𝗳𝗳𝗶𝗰𝗶𝗲𝗻𝘁 𝗡𝗲𝘁𝘄𝗼𝗿𝗸𝘀

<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/particle1331/steepest-ascent" data-color-scheme="no-preference: dark; light: light; dark: dark;" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star particle1331/steepest-ascent on GitHub">Star</a>
<!-- Place this tag in your head or just before your close body tag. -->
<script async defer src="https://buttons.github.io/buttons.js"></script>


This is a collection of notebooks on machine learning theory and engineering. 
Each notebook should run end-to-end after installing the [`inefficient_networks`](https://github.com/particle1331/inefficient-networks/tree/master/src/inefficient_networks) package. This package provides utilities for downloading datasets from Kaggle and reducing boilerplate code in setting up the coding environment. Datasets used are from [Kaggle](https://www.kaggle.com/datasets) or from an open-source library such as
[`torchvision`](https://pytorch.org/vision/stable/index.html) and [`tensorflow_datasets`](https://www.tensorflow.org/datasets). 


The name of this collection is inspired by the following study:


```
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
**Evolution of solution discovery for members of the two groups.** The efficient network converged on a small set of solutions, whereas individuals in the inefficient network explored a greater diversity of solutions, and eventually converged on the best solution. [Fig. 3 in study]
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