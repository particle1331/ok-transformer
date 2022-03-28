# 𝗜𝗻𝗲𝗳𝗳𝗶𝗰𝗶𝗲𝗻𝘁 𝗡𝗲𝘁𝘄𝗼𝗿𝗸𝘀

<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/particle1331/steepest-ascent" data-color-scheme="no-preference: dark; light: light; dark: dark;" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star particle1331/steepest-ascent on GitHub">Star</a>
<!-- Place this tag in your head or just before your close body tag. -->
<script async defer src="https://buttons.github.io/buttons.js"></script>

This is a collection of notebooks on machine learning theory and engineering. 
Each notebook should run end-to-end after installing the [`inefficient_networks`](https://github.com/particle1331/inefficient-networks/tree/dev/package) utilities library (see below).
Datasets come from [Kaggle](https://www.kaggle.com/) or as part of some open-source library such as
[`torchvision`](https://pytorch.org/vision/stable/index.html) and [`tensorflow_datasets`](https://www.tensorflow.org/datasets). 

The name of this collection comes from the following study which changed my mental model on the pursuit of "best" solutions. The results suggest that one should pursue unique and novel solutions to problems &mdash; even if these solutions may seem to be suboptimal at the first iterations.

```text
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
[Fig. 3 of the above study]. Evolution of solution discovery for members of the two groups. The efficient network converged on a small set of solutions, whereas individuals in the inefficient network explored a greater diversity of solutions, and eventually converged on the best solution.
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

Note that [`requirements.txt`](https://github.com/particle1331/inefficient-networks/blob/dev/package/requirements/requirements.txt) has TensorFlow commented out &mdash; this only serves to specify the version used in the notebooks. How to install TensorFlow depends on the operating system and available hardware, so I leave this task to the reader.


## References 

```{bibliography}
```