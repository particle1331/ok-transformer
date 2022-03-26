#!/usr/bin/env python
# coding: utf-8

# # Activation Functions

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

# In this notebook, we will take a closer look at popular activation functions and investigate their effect on optimization properties in neural networks.
# Activation functions are a crucial part of deep learning models as they add the nonlinearity to neural networks.
# There is a great variety of activation functions in the literature, and some are more beneficial than others.
# The goal of this tutorial is to show the importance of choosing a good activation function (and how to do so), and what problems might occur if we don't.

# ```{margin}
# ⚠️ **Attribution:** This notebook builds on [Tutorial 3: Activation Functions](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html) by translating all PyTorch code to TensorFlow 2, and modifying or adding to the discussion. The original tutorial is part of a lecture series on Deep Learning at the University of Amsterdam. The full list of tutorials can be found [here](https://uvadlc-notebooks.rtfd.io). 
# ```

# In[1]:


import tensorflow.keras.optimizers as optim
import tensorflow_datasets as tfds
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices())


# In[2]:


import pathlib
import json
import math
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Path to datasets
DATASET_PATH = pathlib.Path(os.getcwd()).parents[2] / "data"
DATASET_PATH.mkdir(exist_ok=True, parents=True)


# ## Commonly used activations

# As a first step, we will implement some common activation functions by ourselves. Of course, most of them can also be found in the `tf.keras.layers` package. However, we'll write our own functions here for better understanding and insights.
# 
# For an easier time of comparing various activation functions, we start with defining a base class from which all our future modules will inherit. Every activation function will be a Keras layer so that we can integrate them nicely in a network. Note that every Keras layer has a `get_config()` which we can update to include parameters for some activation functions.

# In[3]:


class ActivationFunction(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

# Test
a = ActivationFunction()
a.get_config()


# Next, we implement two of the "oldest" activation functions that are still commonly used for various tasks: sigmoid and tanh. 
# Both the sigmoid and tanh activation can be also found as Keras functions (`tf.keras.activations.sigmoid`, `tf.keras.activations.tanh`). 
# Here, we implement them by hand:

# In[4]:


class Sigmoid(ActivationFunction):
    def call(self, x):
        return 1 / (1 + tf.math.exp(-x))

    
class Tanh(ActivationFunction):
    def call(self, x):
        exp_x, exp_neg_x = tf.math.exp(x), tf.math.exp(-x)
        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)


# Another popular activation function that has allowed the training of deeper networks, is the Rectified Linear Unit (ReLU). Despite its simplicity of being a piecewise linear function, ReLU has one major benefit compared to sigmoid and tanh: a strong, stable gradient for a large range of values. Based on this idea, a lot of variations of ReLU have been proposed, of which we will implement the following three: LeakyReLU, ELU, and Swish. 
# 
# LeakyReLU replaces the zero settings in the negative part with a smaller slope to allow gradients to flow also in this part of the input. Similarly, ELU replaces the negative part with an exponential decay. The third, most recently proposed activation function is Swish, which is actually the result of a large experiment with the purpose of finding the “optimal” activation function. Compared to the other activation functions, Swish is both smooth and non-monotonic (i.e. contains a change of sign in the gradient). This has been shown to prevent dead neurons as in standard ReLU activation, especially for deep networks. If interested, a more detailed discussion of the benefits of Swish can be found in [this paper](https://arxiv.org/abs/1710.05941). Note that out of all activations, only Swish has maximum gradient greater than 1.
# 
# Let’s implement the four activation functions below:

# In[5]:


class ReLU(ActivationFunction):
    def call(self, x):
        return tf.where(x > 0, x, [0])
    

class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = tf.Variable(alpha, trainable=False)
    
    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha})
        return config

    def call(self, x):
        return tf.where(x > 0, x, x * self.get_config()["alpha"])


class ELU(ActivationFunction):
    def call(self, x):
        # For some reason `tf.exp(x) - c` is not registered in the GPU
        return tf.where(x > 0, x, tf.exp(x) + (-1.0))
    

class Swish(ActivationFunction):
    def call(self, x):
        return x * tf.keras.activations.sigmoid(x)


# For later usage, we summarize all our activation functions in a dictionary mapping the name to the class object. In case you implement a new activation function by yourself, add it here to include it in future comparisons as well:

# In[6]:


act_fn_by_name = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "leakyrelu": LeakyReLU,
    "elu": ELU,
    "swish": Swish,
}


# ### Plotting activation values
# 
# To get an idea of what each activation function actually does, we will visualize them in the following. 
# Next to the actual activation value, the gradient of the function is an important aspect as it is crucial for optimizing the neural network. 
# TensorFlow allows us to compute the gradients `∂z/∂x` using `tape.gradient(z, x)`. The resulting gradients has the same shape as `x`. Below we will let `x` be all trainable parameters of the network, which can be huge. However, TensorFlow is able to calculate everything in an efficient manner with a single call (instead of calling `.gradient` for each weight and persisting the tape).

# In[7]:


def get_grads(act_fn, x):
    """Compute gradient of act_fn with respect to x."""
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = act_fn(x)
        z = tf.reduce_sum(y)

    # Trick: ∂(y1 + y2)/∂x1 = ∂y1/∂x1 stored in x1.
    return tape.gradient(z, x)


# Now we can visualize all our activation functions including their gradients:

# In[8]:


def plot_activation(act_fn, ax, x):
    """Plot (x, act_fn(x)) on axis object ax."""

    # Get output and gradients from input space x.
    y = act_fn(x)
    y_grads = get_grads(act_fn, x)

    # Convert to numpy for plotting
    x, y, y_grads = x.numpy(), y.numpy(), y_grads.numpy()
    
    # Plotting
    ax.plot(x, y, linewidth=2, color='red', label="ActFn")
    ax.plot(x, y_grads, linewidth=1, color='black', label="Gradient")
    ax.set_title(act_fn.name)
    ax.legend()
    ax.grid()
    ax.set_ylim(-1.5, x.max())


# Initialize activations in a list
act_fns = [act_fn() for act_fn in act_fn_by_name.values()]

# Plotting
x = tf.reshape(tf.linspace(-5, 5, 1000), (-1, 1))
rows = math.ceil(len(act_fns) / 2.0)
fig, ax = plt.subplots(rows, 2, figsize=(12, rows*5), dpi=300)
for i, act_fn in enumerate(act_fns):
    plot_activation(act_fn, ax[divmod(i, 2)], x) # divmod(m, n) = m // n, m % n

fig.subplots_adjust(hspace=0.3)
plt.show()


# ## Analyzing the effect of activation functions

# After implementing and visualizing the activation functions, we are aiming to gain insights into their effect. 
# We do this by using a simple neural network trained on [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) and examine various aspects of the model, including the performance and gradient flow.

# ### Base MLP network

# Let's set up a base neural network. The chosen network views the images as 1D tensors and pushes them through a sequence of linear layers and a specified activation function. Hence, an MLP with flattened images as input. Each neuron in the next layer has a weight vector that acts as a filter for the whole image. Projecting the image in this filter which returns a real number measuring the degree of projection. Thus, its important to normalize each image. The numbers form a vector which is processed in the same manner in the next layer. Feel free to experiment with other network architectures.

# In[9]:


def base_net(act_fn, num_classes=10, hidden_sizes=(512, 256, 256, 128)):
    """Return an initialized MLP network with dense hidden layers with activation
    `act_fn` and width in `hidden_sizes` ordered such that index zero is nearest 
    the input layer, and a final linear layer (logits) of width `num_classes`."""

    model = tf.keras.Sequential()

    # Add hidden layers with activation
    for j in range(len(hidden_sizes)):
        model.add(tf.keras.layers.Dense(
            units=hidden_sizes[j], 
            kernel_initializer=tf.keras.initializers.HeNormal(seed=RANDOM_SEED)
        ))
        model.add(act_fn())

    # Add logit linear layer
    model.add(tf.keras.layers.Dense(units=num_classes))
    return model


# Testing: 

# In[10]:


model = base_net(lambda: LeakyReLU())
model.build(input_shape=(None, 784))
model.summary()

x = tf.random.normal(shape=(1, 784))
tf.print(model(x))


# The trainable parameters of the network can be accessed as follows. This skips the parameters of the LeakyReLU activations which are nontrainable.

# In[11]:


print("All trainable variables:")
for v in model.trainable_variables:
    tf.print(f"  {str(v.name):<15}\t {str(v.shape):<15}")


# ### Preprocessing FashionMNIST

# [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) is a more complex version of MNIST and contains black-and-white images of clothes instead of digits. The 10 classes include trousers, coats, shoes, bags and more. To load this dataset, we will make use of `tensorflow_datasets`.

# In[12]:


FMNIST, FMNIST_info = tfds.load('fashion_mnist', data_dir=DATASET_PATH, with_info=True, shuffle_files=False)
print(FMNIST_info)


# In[13]:


# Transformations applied on each image. 
def transform_image(image):
    return tf.reshape(tf.keras.layers.Rescaling(1./255)(image), (-1,))

train_ds, test_ds = FMNIST['train'], FMNIST['test']
train_ds = train_ds.map(lambda x: (transform_image(x['image'][:, :, 0]), x['label']))
test_ds = test_ds.map(lambda x: (transform_image(x['image'][:, :, 0]), x['label']))


# Let's visualize a few images to get an impression of the data.

# In[14]:


images, labels = next(iter(train_ds.batch(16)))
class_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

fig = plt.figure(figsize=(8, 8))
for i in range(16):
    image = images[i].numpy()
    label = labels[i]
    ax = fig.add_subplot(4, 4, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image.reshape(28, 28), cmap="gray")
    ax.set_title(class_map[label.numpy()], size=15)
    
plt.tight_layout()
plt.show()


# ### Gradient flow after initialization
# 
# As mentioned previously, one important aspect of activation functions is how they propagate gradients through the network. Imagine we have a very deep neural network with more than 50 layers. The gradients for the input layer, i.e. the very first layer, have passed >50 times the activation function, but we still want them to be of a reasonable size. If the gradient through the activation function is in expectation considerably smaller than 1, our gradients will vanish until they reach the input layer. If the gradient through the activation function is larger than 1, the gradients exponentially increase and might explode. These are known as the problems of **vanishing** and **exploding gradients**.

# In[35]:



def plot_gradient_histogram(model, act_fn_name, color="C0"):
    """Plot histogram of gradients after one backprop from a batch of inputs."""

    # Get one batch of images
    small_loader = train_ds.batch(256)
    images, labels = next(iter(small_loader))

    # Pass the batch through the network, and calculate the gradients for the weights
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape(persistent=True) as tape:
        preds = model(images)
        loss = loss_fn(labels, preds)
    
    # Exclude the bias to reduce the number of plots
    grads_dict = {}
    for layer_index, layer in enumerate(model.layers):
        grads = tape.gradient(loss, layer.variables)
        for j in range(len(layer.variables)):
            w = layer.variables[j]
            if (not w.trainable) or ("bias" in w.name):
                continue
            grads_dict[f"({layer_index}) {w.name.split('/')[0]}"] = grads[j].numpy().reshape(-1)

    # Plotting
    columns = len(grads_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns*3.5, 2.5), dpi=300)
    fig_index = 0

    for key in sorted(grads_dict.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(data=grads_dict[key], bins=30, ax=key_ax, color=color, kde=True)
        key_ax.set_xlabel("Grad magnitude")
        key_ax.set_title(key)
        fig_index += 1
    
    fig.suptitle(f"Gradient magnitude distribution (weights) for activation function {act_fn_name}", fontsize=14, y=1.05)
    fig.subplots_adjust(wspace=0.45)
    plt.legend([])
    plt.show()
    plt.close()


# To get a feeling of how every activation function influences the gradients, we can look at a freshly initialized network and measure the gradients for each parameter for a batch of 256 images. That is, we pass 256 images, backpropagate gradients based on the available labels for this images, then look at the histogram of gradient values. In particular, we look for vanishing and exploding gradients. 

# In[36]:


# Seaborn prints warnings if histogram has small values. We can ignore them for now
import warnings
warnings.filterwarnings('ignore')

# Create a plot for every activation function
# Setting the seed ensures that we have the same weight initialization
for i, act_fn_name in enumerate(act_fn_by_name):
    act_fn = act_fn_by_name[act_fn_name]
    net_actfn = base_net(act_fn=act_fn)
    plot_gradient_histogram(net_actfn, act_fn_name=act_fn_name, color=f"C{i}")


# The sigmoid activation function shows a clearly undesirable behavior. While the gradients for the output layer are very large with up to 0.1, the input layer has the lowest gradient norm across all activation functions with only 1e-4. This is due to its small maximum gradient of 1/4, and finding a suitable learning rate across all layers is not possible in this setup.
# All the other activation functions show to have similar gradient norms across all layers. Interestingly, the ReLU activation has a spike around 0 which is caused by its zero-part on the left, and dead neurons (we will take a closer look at this later on).
# 
# Note that additionally to the activation, the weight and bias initialization can be crucial. By default, TensorFlow uses the Glorot uniform initialization for linear layers optimized for sigmoid activations. Note that in our implementation we used Kaiming normal initialization which is optimized for ReLU activations. In *Optimization and Initialization*, we will take a closer look at initialization, but assume for now that the Kaiming initialization works for all activation functions reasonably well.

# ### Training with different activations
# 
# Next, we want to train our model with different activation functions on FashionMNIST and compare the gained performance. All in all, our final goal is to achieve the best possible performance on a dataset of our choice. 
# Therefore, we write a training loop in the next cell including a validation after every epoch and a final test on the best model:

# In[17]:


def train_model(model, max_epochs=50, batch_size=256):
    """
    Train a model on the training set of FashionMNIST.
    Inputs:
        model - An instance of our base MLP network.
        max_epochs (int) - Training budget, i.e. max no. of epochs to train networks.
        batch_size (int) - Size of batches used in training when loading train data.
    """

    # Defining optimizer, loss, metrics, and early stopping callback
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True)
    
    # Build and compile
    model.build(input_shape=(None, 784))
    model.compile(
        optimizer=optimizer,
        loss=loss, 
        metrics=[metrics]
    )

    # Recall shuffle, batch, repeat pattern to create epochs
    train_loader = train_ds.shuffle(buffer_size=1000)
    train_loader = train_loader.batch(batch_size=batch_size, drop_remainder=True)
    train_loader = train_loader.repeat()
    train_loader = train_loader.prefetch(buffer_size=batch_size)    # Prepare next elements 
                                                                    # while current is preprocessed. 
                                                                    # Trades off latency with memory.

    test_loader = test_ds.shuffle(buffer_size=4096)
    test_loader = test_loader.batch(2048)
    test_loader = test_loader.repeat()

    # Train model
    steps_per_epoch = np.ceil(len(train_ds)/batch_size)
    history = model.fit(
        train_loader, 
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch, 
        verbose=0,
        validation_data=test_loader,
        validation_steps=1,
        callbacks=[early_stopping],
    )
    
    # Return history and final test accuracy
    test_acc = model.evaluate(test_ds.batch(1), verbose=0)
    return history, test_acc


# Iterating over all activation functions:

# In[18]:


results = {}
_, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

for act_fn_name in act_fn_by_name.keys():
    model = base_net(lambda: act_fn_by_name[act_fn_name]())
    history, test_acc = train_model(model, max_epochs=5)
    results[act_fn_name] = test_acc[1]
    
    # Plotting train loss
    n = len(history.history['loss'])
    ax[0, 0].set_title("Train loss")
    ax[0, 0].plot(range(n), history.history['loss'], label=act_fn_name)
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    ax[1, 0].set_title("Valid loss")
    ax[1, 0].plot(range(n), history.history['val_loss'], label=act_fn_name)
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Plotting accuracy
    ax[0, 1].set_title("Train accuracy")
    ax[0, 1].plot(range(n), history.history['sparse_categorical_accuracy'], label=act_fn_name)
    ax[0, 1].legend()
    ax[0, 1].grid(True)
    ax[1, 1].set_title("Valid accuracy")
    ax[1, 1].plot(range(n), history.history['val_sparse_categorical_accuracy'], label=act_fn_name)
    ax[1, 1].legend()
    ax[1, 1].grid(True)


# Repeating the same experiment but not plotting sigmoid and with more max epochs:

# In[19]:


results = {}
trained_models = {}
_, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

for act_fn_name in act_fn_by_name.keys():
    model = base_net(lambda: act_fn_by_name[act_fn_name]())
    history, test_acc = train_model(model, max_epochs=50)
    results[act_fn_name] = test_acc[1] # new_model.metrics_names -> ['loss', 'sparse_categorical_accuracy']
    trained_models[act_fn_name] = model
    
    if act_fn_name != "sigmoid":
        # Plotting train loss
        n = len(history.history['loss'])
        ax[0, 0].set_title("Train loss")
        ax[0, 0].plot(range(n), history.history['loss'], label=act_fn_name)
        ax[0, 0].legend()
        ax[0, 0].grid(True)
        ax[1, 0].set_title("Valid loss")
        ax[1, 0].plot(range(n), history.history['val_loss'], label=act_fn_name)
        ax[1, 0].legend()
        ax[1, 0].grid(True)

        # Plotting accuracy
        ax[0, 1].set_title("Train accuracy")
        ax[0, 1].plot(range(n), history.history['sparse_categorical_accuracy'], label=act_fn_name)
        ax[0, 1].legend()
        ax[0, 1].grid(True)
        ax[1, 1].set_title("Valid accuracy")
        ax[1, 1].plot(range(n), history.history['val_sparse_categorical_accuracy'], label=act_fn_name)
        ax[1, 1].legend()
        ax[1, 1].grid(True)


# The final test accuracies are shown in the following table (these are from the best weights since we set `restore_best_weights` to `True` in our early stopping callback):

# In[20]:


import pandas as pd
results_df = pd.DataFrame({k: [results[k]] for k in results.keys()})
results_df


# Not surprisingly, the model using the sigmoid activation function has relatively bad performance. This is because of the low magnitudes of the gradients on layers near the input. All the other activation functions gain 
# similar performance.
# 
# To have a more accurate conclusion, we would have to train the models for multiple seeds and look at the averages.
# However, the "optimal" activation function also depends on many other factors (hidden sizes, number of layers, type of layers, task, dataset, optimizer, learning rate, etc.) so that a thorough grid search would not be useful in our case.
# 
# In the literature, activation functions that have shown to work well with deep networks are all types of ReLU functions we experiment with here, with small gains for specific activation functions in specific networks.

# ### Distribution of activation values

# After we have trained the models, we can look at the actual activation values that find inside the model. For instance, how many neurons are set to zero in ReLU? Where do we find most values in Tanh?
# To answer these questions, we can write a simple function which takes a trained model, applies it to a batch of images, and plots the histogram of the activations inside the network over the input batch. Note that we only look at the activations in the hidden layers excluding the logits layer (which is a linear layer).

# In[21]:


def plot_activations_histogram(model, act_fn_name, color="C0"):
    # We need to manually loop through the layers to save all activations
    small_loader = train_ds.batch(batch_size=1024)
    images, labels = next(iter(small_loader))
    activations = {}
    x = images
    for layer_index, layer in enumerate(model.layers[:-1]):
        x = layer(x)
        activations[layer_index] = x.numpy().reshape(-1)

    # Plotting
    columns = 4
    rows = math.ceil(len(activations)/columns)
    fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5), dpi=300)
    fig_index = 0
    
    for key in activations: # key := layer_index
        key_ax = ax[divmod(fig_index, columns)]
        sns.histplot(data=activations[key], bins=50, ax=key_ax, color=color, kde=True, stat="density")
        key_ax.set_title(f"({key}): {model.layers[key].name}")
        fig_index += 1
        
    fig.suptitle(f"Activation distribution for activation function {act_fn_name}", fontsize=12)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    return fig, ax


# Plotting activations distribution of trained models:

# In[22]:


i = 0
for act_fn_name in act_fn_by_name.keys():
    plot_activations_histogram(trained_models[act_fn_name], act_fn_name, color=f"C{i}")
    i += 1


# As the model with sigmoid activation was not able to train properly, the activations are also less informative.
# 
# The tanh shows a more diverse behavior. While for the input layer we experience a larger amount of neurons to be close to -1 and 1, where the gradients are close to zero, the activations in the two consecutive layers are closer to zero. This is probably because the input layers look for specific features in the input image, and the consecutive layers combine those together. The activations for the last layer are again more biased to the extreme points because the classification layer can be seen as a weighted average of those values (the gradients push the activations to those extremes).
# 
# 
# The ReLU has a strong peak at 0, as we initially expected. The effect of having no gradients for negative values is that the network does not have a Gaussian-like distribution after the linear layers, but a longer tail towards the positive values. The LeakyReLU shows a very similar behavior while ELU follows again a more Gaussian-like distribution. The Swish activation seems to lie in between, although it is worth noting that Swish uses significantly higher values than other activation functions.
# 
# 
# As all activation functions show slightly different behavior although obtaining similar performance for our simple network, it becomes apparent that the selection of the "optimal" activation function really depends on many factors, and is not the same for all possible networks.

# ## Finding dead neurons in ReLU networks

# One known drawback of the ReLU activation is the occurrence of "dead neurons", i.e. neurons with no gradient for any training input.
# The issue of dead neurons is that as no gradient flows across the layer, and we cannot train the parameters of this neuron in the previous layer to obtain output values besides zero.
# 
# For dead (or dying) neurons to happen, the output value of a specific neuron of the linear layer before the ReLU has to be negative for all (or almost all) input images. 
# Note that all gradients are also zero so that the weights will also not update if we introduce no new data. 
# Considering the large number of neurons we have in a neural network, it is not unlikely for this to happen. 
# 
# To get a better understanding of how much of a problem this is, and when we need to be careful, we will measure how many dead neurons different networks have. For this, we implement a function which runs the network on the whole training set and records whether a neuron is "zero" for majority of the data points (99% by default):

# In[23]:


def fraction_dead_neurons(model, threshold=0.99):
    # Initialize counter for each layer
    count_zero_actn = {}
    for index, layer in enumerate(model.layers[:-1]):
        if isinstance(layer, tf.keras.layers.Dense):
            layer_width = layer.get_weights()[1].shape
            count_zero_actn[index+1] = tf.zeros(layer_width)

    train_loader = train_ds.batch(1024)
    for images, _ in train_loader: 
        # Count zero activations during forward pass of one batch
        out = images
        for layer_index in range(len(model.layers[:-1])):
            layer = model.layers[layer_index]
            out = layer(out)
            if isinstance(layer, ActivationFunction):
                count_zero_actn[layer_index] += tf.reduce_sum(tf.cast(tf.abs(out) < 1e-8, tf.float32), axis=0)
    
    # Print fraction of dead neurons in each layer.
    # Dead = fraction of "zero" activations (over train set) exceeds threshold.
    print("Number of dead neurons:")
    for key in count_zero_actn.keys():
        num_dead = tf.reduce_sum(tf.cast((count_zero_actn[key] / len(train_ds)) >= threshold, tf.float32))
        total = count_zero_actn[key].shape[0]
        print(f"  {f'({key})':>4} {model.layers[key].__class__.__name__}: {int(num_dead):3}/{total:3} {f'({(100.0 * num_dead / total):.2f}%)':>8}")


# **Zero network.** Testing with network with all neurons having zero output:

# In[24]:


def zeros_init(net):
    for layer in net.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights([tf.zeros(w.shape) for w in layer.get_weights()])

# Initialize with zero weights
net_zero = base_net(act_fn=lambda: ReLU())
net_zero.build(input_shape=(None, 784))
zeros_init(net_zero)

# Should sum to zero
tf.print(sum([tf.reduce_sum(v) for v in net_zero.variables]))


# In[25]:


fraction_dead_neurons(net_zero) # Should be all dead


# **Untrained network.** Let's measure the number of dead neurons for an untrained network:

# In[26]:


net_relu = base_net(act_fn=lambda: ReLU(), hidden_sizes=[512, 256, 256, 128])
net_relu.build(input_shape=(None, 784))
fraction_dead_neurons(net_relu)


# We see that only a minor amount of neurons are dead, but that they increase with the depth of the layer.
# In the long term, this is not a problem for the small number of dead neurons we have as the input to the neuron's layer is changed due to updates to the weights of lower layers. Such weights can be updated from gradients flowing into lower layers through alternate paths (i.e. across neurons in the same layer that are alive). Therefore, dead neurons can potentially become alive or active again. 

# **Trained network.** How does this look like for a trained network with the same initialization?

# In[27]:


trained_relu = trained_models['relu']
trained_relu.build(input_shape=(None, 784))
fraction_dead_neurons(trained_relu)


# The number of dead neurons indeed decreased in the later layers. However, it should be noted that dead neurons are especially problematic in the input layer since there are no lower layers whose weights can be updated to shift the activations to be predominantly positive. Still, the input data has usually a sufficiently high standard deviation to reduce the risk of dead neurons.

# Finally, we check how the number of dead neurons behaves with increasing layer depth. For instance, let’s take the following 10-layer neural network:

# In[28]:


net_relu = base_net(act_fn=lambda: ReLU(), hidden_sizes=[256, 256, 256, 256, 256, 128, 128, 128, 128, 128])
net_relu.build(input_shape=(None, 784))
fraction_dead_neurons(net_relu)


# The number of dead neurons is significantly higher than before which harms the gradient flow especially in the first iterations. For instance, more than 25% of the neurons in the pre-last layer are dead which creates a considerable bottleneck. Hence, it is advisible to use other nonlinearities like Swish for very deep networks.

# In[29]:


net_swish = base_net(act_fn=lambda: Swish(), hidden_sizes=[256, 256, 256, 256, 256, 128, 128, 128, 128, 128])
net_swish.build(input_shape=(None, 784))
fraction_dead_neurons(net_swish)


# ## Conclusion
# 
# In this notebook, we have reviewed a set of six activation functions (sigmoid, tanh, ReLU, LeakyReLU, ELU, and Swish) in neural networks, and discussed how they influence the gradient distribution across layers. Sigmoid tends to fail deep neural networks as the highest gradient it provides is 0.25 leading to vanishing gradients in early layers. All ReLU-based activation functions have shown to perform well, and besides the original ReLU, do not have the issue of dead neurons. When implementing your own neural network, it is recommended to start with a ReLU-based network and select the specific activation function based on the properties of the network.

# In[ ]:




