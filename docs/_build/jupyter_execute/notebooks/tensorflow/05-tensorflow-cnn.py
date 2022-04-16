#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Networks [TODO]

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Todo&color=red)

# In this notebook, we will learn about
# **convolutional neural networks** (**CNN**) for image classification. We will start
# by discussing the basic building blocks of CNNs, using a bottom-up approach.
# Then, we will take a deeper dive into the CNN architecture and explore how to
# implement CNNs in TensorFlow.

# ## Building blocks of CNNs

# We will discuss the broader concept of CNNs and why
# convolutional architectures are often described as "feature extraction layers." Then,
# we will delve into the theoretical definition of the type of convolution operation that
# is commonly used in CNNs and walk through examples for computing convolutions
# in one and two dimensions.

# ### Understanding CNNs and feature hierarchies

# Certain types of multilayer NNs, and in particular, deep convolutional NNs,
# construct a so-called feature hierarchy by combining the low-level features in a
# layer-wise fashion to form high-level features. For example, if we're dealing with
# images, then low-level features, such as edges and blobs, are extracted from the
# earlier layers, which are combined together to form high-level features. These high-level features can form more complex shapes, such as the general contours of objects. 
# For this reason, it's common
# to consider CNN layers as feature extractors: the early layers (those right after the
# input layer) extract low-level features from raw data, and the later layers (often fully
# connected layers like in a MLP) use these features to predict
# a continuous target value or class label.

# A CNN computes **feature maps** from an input
# image, where each element comes from a local patch of pixels in the input image. This local patch of pixels is referred to as the local receptive field. CNNs will usually
# perform very well on image-related tasks, and this is largely due to two important ideas:
# 
# * **Sparse connectivity**: A single element in the feature map is connected to only
# a small patch of pixels.
# 
# * **Parameter-sharing**: The same weights are used for different patches of the
# input image.

# Note that this is very different from connecting to the whole
# input image as in the case of perceptrons. As a direct consequence of these two ideas, replacing a conventional, fully connected
# MLP with a convolution layer substantially decreases the number of weights
# in the network and we will see an improvement in the ability to capture
# salient features. In the context of image data, it makes sense to assume that nearby
# pixels are typically more relevant to each other than pixels that are far away from
# each other.

# ### Discrete convolutions

# A **discrete convolution** (or simply **convolution**) is a fundamental operation in
# a CNN. Therefore, it's important to understand how this operation works. In
# this section, we will cover the mathematical definition and discuss some of the
# naive algorithms to compute convolutions of vectors and matrices (one and two-dimensional tensors).

# #### Discrete convolutions in one dimension

# A discrete convolution for two vectors $\boldsymbol x$ (signal) and $\boldsymbol w$ (kernel) is defined as follows:
# 
# $$\boldsymbol y = \boldsymbol x * \boldsymbol w \quad\rightarrow\quad y_i = \sum_{k=-\infty}^{+\infty} x_{i-k} w_k$$
# 
# 

# 
