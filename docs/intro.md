# 𝗜𝗻𝗲𝗳𝗳𝗶𝗰𝗶𝗲𝗻𝘁 𝗡𝗲𝘁𝘄𝗼𝗿𝗸𝘀

<!-- Place this tag where you want the button to render. -->
<a class="github-button" href="https://github.com/particle1331/steepest-ascent" data-color-scheme="no-preference: dark; light: light; dark: dark;" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star particle1331/steepest-ascent on GitHub">Star</a>
<!-- Place this tag in your head or just before your close body tag. -->
<script async defer src="https://buttons.github.io/buttons.js"></script>

This is a collection of notebooks on machine learning theory and engineering. 
Mostly of things I study and want to reproduce, or rewrite to reflect my own understanding 
and way of doing things.
These notebooks should run end-to-end after installing the [`inefficient_networks`](https://github.com/particle1331/inefficient-networks/tree/master/src/inefficient_networks) helper package. 

Note that there is really no strong dependence on this package &mdash; this just consists of utilities for automatically creating directories and downloading [Kaggle datasets](https://www.kaggle.com/datasets).
Hence, you can easily modify the code to eliminate this dependency when running individual notebooks.

The name of this collection is a reference to the following study:

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
[Fig. 3 in the above study]. **Evolution of solution discovery for members of the two groups.** The efficient network converged on a small set of solutions, whereas individuals in the inefficient network explored a greater diversity of solutions, and eventually converged on the best solution.
```

## Installation

The package [`inefficient_networks`](https://github.com/particle1331/inefficient-networks/tree/dev/package) provides utilities for downloading data from Kaggle and a [`config`](https://github.com/particle1331/inefficient-networks/blob/dev/package/inefficient_networks/config.py) object to reduce boilerplate code in setting up the coding environment. 
To install:

```
$ git clone git@github.com:particle1331/inefficient-networks.git
$ cd inefficient-networks
$ cd src
$ pip install -e .
```

Note that TensorFlow has been commented out in [`requirements.txt`](https://github.com/particle1331/inefficient-networks/blob/dev/package/requirements/requirements.txt). This only serves to specify the version used in the notebooks &mdash; you would have to install it for your specific OS and hardware.

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