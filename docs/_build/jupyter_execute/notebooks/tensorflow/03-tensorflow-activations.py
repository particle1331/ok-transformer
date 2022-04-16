#!/usr/bin/env python
# coding: utf-8

# # Activation Functions

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

# In this notebook, we will take a closer look at popular activation functions and investigate their effect on optimization properties in neural networks.
# Activation functions are a crucial part of deep learning models as they add the nonlinearity to neural networks.
# There is a great variety of activation functions in the literature, and some are more beneficial than others.
# The goal of this tutorial is to show the importance of choosing a good activation function (and how to do so), and what problems might occur if we don't.

# ```{margin}
# ⚠️ **Attribution:** This notebook builds on [Tutorial 3: Activation Functions](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial3/Activation_Functions.html). The original tutorial is written in PyTorch and is part of a lecture series on Deep Learning at the University of Amsterdam. The original tutorials are released under [MIT License](https://github.com/phlippe/uvadlc_notebooks/blob/master/LICENSE.md).
# ```

# In[1]:


import math
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as kr

from inefficient_networks import utils
from inefficient_networks.config import config 

config.set_matplotlib()
config.set_tensorflow_seeds(42)
config.set_ignore_warnings()
print(config.list_tensorflow_devices())
print(tf.__version__)


# ## Commonly used activations

# As a first step, we will implement some common activation functions by ourselves. Of course, most of them can also be found in the `kr.layers` module. However, we'll write our own functions here for better understanding and insights.
# 
# For an easier time of comparing various activation functions, we start with defining a base class from which all our future modules will inherit. Every activation function will be a Keras layer so that we can integrate them nicely in a network. Note that every Keras layer has a `get_config()` which we can update to include parameters for some activation functions.

# In[2]:


class ActivationFunction(kr.layers.Layer):
    def __init__(self):
        super().__init__()

# Test
a = ActivationFunction()
a.get_config()


# Next, we implement two of the "oldest" activation functions that are still commonly used for various tasks: sigmoid and tanh. 
# Both the sigmoid and tanh activation can be also found as Keras functions (`kr.activations.sigmoid`, `kr.activations.tanh`). 
# Here, we implement them by hand:

# In[3]:


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

# In[4]:


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
        return x * kr.activations.sigmoid(x)


# For later usage, we summarize all our activation functions in a dictionary mapping the name to the class object. In case you implement a new activation function by yourself, add it here to include it in future comparisons as well:

# In[5]:


activation_by_name = {
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

# In[6]:


def get_grads(activation, x):
    """Compute gradient of activation with respect to x."""
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = activation_by_name[activation]()(x)
        z = tf.reduce_sum(y)

    # Trick: ∂(y1 + y2)/∂x1 = ∂y1/∂x1 stored in x1.
    return tape.gradient(z, x)


# Now we can visualize all our activation functions including their gradients:

# In[7]:


def plot_activation(activation, ax, x):
    """Plot (x, activation(x)) on axis object ax."""

    # Get output and gradients from input space x.
    y = activation_by_name[activation]()(x)
    y_grads = get_grads(activation, x)

    # Convert to numpy for plotting
    x, y, y_grads = x.numpy(), y.numpy(), y_grads.numpy()
    
    # Plotting
    ax.plot(x, y, linewidth=2, color='red', label="Activation")
    ax.plot(x, y_grads, linewidth=1, color='black', label="Gradient")
    ax.set_title(activation)
    ax.legend()
    ax.grid()
    ax.set_ylim(-1.5, x.max())


# Plotting
x = tf.reshape(tf.linspace(-5, 5, 1000), (-1, 1))
rows = math.ceil(len(activation_by_name.keys()) / 2.0)
fig, ax = plt.subplots(rows, 2, figsize=(12, rows*5))
for i, activation in enumerate(activation_by_name.keys()):
    plot_activation(activation, ax[divmod(i, 2)], x) # divmod(m, n) = m // n, m % n

fig.subplots_adjust(hspace=0.3)
plt.show()


# ## Analyzing the effect of activation functions

# After implementing and visualizing the activation functions, we are aiming to gain insights into their effect. 
# We do this by using a simple neural network trained on [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) and examine various aspects of the model, including the performance and gradient flow.

# ### Preliminaries

# **Base network.** Let's set up a base neural network. The chosen network views the images as 1D tensors and pushes them through a sequence of linear layers and a specified activation function. Hence, an MLP with flattened images as input. Each neuron in the next layer has a weight vector that acts as a filter for the whole image. Projecting the image in this filter which returns a real number measuring the degree of projection. Thus, its important to normalize each image. The numbers form a vector which is processed in the same manner in the next layer. Feel free to experiment with other network architectures.

# In[8]:


def base_network(
    activation,
    num_classes=10,
    hidden_sizes=(512, 256, 256, 128)
):
    """Return a fully-connected network with given activation and layer widths."""

    # Add dense hidden layers + activation layer
    model = kr.Sequential()
    for j in range(len(hidden_sizes)):
        model.add(kr.layers.Dense(hidden_sizes[j]))
        model.add(activation_by_name[activation]())

    # Add linear logit layer
    model.add(kr.layers.Dense(units=num_classes))
    
    return model


# Testing: 

# In[9]:


model = base_network(activation='leakyrelu')
model.build(input_shape=(None, 784))
model.summary()

x = tf.random.normal(shape=(1, 784))
tf.print(model(x))


# The trainable parameters of the network can be accessed as follows. This skips the parameters of the LeakyReLU activations which are nontrainable.

# In[10]:


print("All trainable variables:")
for v in model.trainable_variables:
    tf.print(f"  {str(v.name):<15}\t {str(v.shape):<15}")


# **Dataset.** [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) is a more complex version of MNIST and contains black-and-white images of clothes instead of digits. The 10 classes include trousers, coats, shoes, bags and more. To load this dataset, we will make use of `tensorflow_datasets`.

# In[11]:


FMNIST, FMNIST_info = tfds.load(
    'fashion_mnist', 
    data_dir=config.DATASET_DIR, 
    with_info=True, 
    shuffle_files=False
)
print(FMNIST_info)


# In[12]:


# Transformations applied on each image. 
def transform_image(image):
    """Rescale image linearly from [0, 255] to [0, 1]."""
    return tf.reshape(kr.layers.Rescaling(1./255)(image), (-1,))


train_dataset, test_dataset = FMNIST['train'], FMNIST['test']
train_dataset = train_dataset.map(lambda x: (transform_image(x['image'][:, :, 0]), x['label']))
test_dataset = test_dataset.map(lambda x: (transform_image(x['image'][:, :, 0]), x['label']))

# Create fixed batch for all subsequent experiments
fixed_batch = next(iter(train_dataset.batch(4096)))


# Let's visualize a few images to get an impression of the data.

# In[13]:


images, labels = fixed_batch
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

# In[14]:


def plot_gradient_distribution(model, activation_name, color="C0"):
    """Plot histogram of gradients after one backprop from a batch of inputs."""

    # Get one batch of images
    images, labels = fixed_batch

    # Pass the batch through the network, and calculate the gradients for the weights
    loss_fn = kr.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape(persistent=True) as tape:
        preds = model(images)
        loss = loss_fn(labels, preds)
    
    # Exclude the bias to reduce the number of plots
    grads_dict = {}
    for i, layer in enumerate(model.layers):
        grads = tape.gradient(loss, layer.variables)

        # Get kernel weights
        for j in range(len(layer.variables)):
            w = layer.variables[j]
            if (not w.trainable) or ("bias" in w.name):
                continue
            grads_dict[f"({i}) {w.name.split('/')[0]}"] = grads[j].numpy().reshape(-1)

    # Plotting
    columns = len(grads_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns*3.5, 2.5))
    fig_index = 0

    for key in sorted(grads_dict.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(data=grads_dict[key], bins=30, stat='count', ax=key_ax, color=color, kde=True)
        key_ax.set_xlabel("Grad magnitude")
        key_ax.set_title(key)
        fig_index += 1
    
    fig.suptitle(f"Gradient magnitude distribution (weights) for activation function {activation_name}", fontsize=14, y=1.10)
    fig.subplots_adjust(wspace=0.45)
    plt.legend([])
    plt.show()
    plt.close()


# To get a feeling of how every activation function influences the gradients, we can look at a freshly initialized network and measure the gradients for each parameter for a batch of 256 images. That is, we pass 256 images, backpropagate gradients based on the available labels for this images, then look at the histogram of gradient values. In particular, we look for vanishing and exploding gradients. 

# In[15]:


import warnings
warnings.filterwarnings('ignore')

# Create a plot for every activation function. Setting the 
# seed ensures that we have the same weight initialization.
for i, activation in enumerate(activation_by_name.keys()):
    model = base_network(activation=activation)
    plot_gradient_distribution(model, activation_name=activation, color=f"C{i}")


# The sigmoid activation function shows a clearly undesirable behavior. While the gradients for the output layer are very large with up to 0.1, the input layer has the lowest gradient norm across all activation functions with only 1e-4. This is due to its small maximum gradient of 1/4, and finding a suitable learning rate across all layers is not possible in this setup.
# All the other activation functions show to have similar gradient norms across all layers. Interestingly, the ReLU activation has a spike around 0 which is caused by its zero-part on the left, and dead neurons (we will take a closer look at this later on).
# 
# Note that additionally to the activation, the weight and bias initialization can be crucial. By default, TensorFlow uses the Glorot uniform initialization for linear layers optimized for sigmoid activations. Note that in our implementation we used Kaiming normal initialization which is optimized for ReLU activations. In [*Initialization and Optimization*](https://particle1331.github.io/inefficient-networks/notebooks/tensorflow/04-tensorflow-optim-init.html), we will take a closer look at initialization, but assume for now that the Kaiming initialization works for all activation functions reasonably well.

# ### Training with different activations
# 
# Next, we want to train our model with different activation functions on FashionMNIST and compare the gained performance. All in all, our final goal is to achieve the best possible performance on a dataset of our choice. 
# Therefore, we write a training loop in the next cell including a validation after every epoch and a final test on the best model:

# In[16]:


def train_model(model, train_loader, test_loader, batch_size=256, max_epochs=50):
    """Train model on FashionMNIST. Restore best weights on early stop."""

    # Defining optimizer, loss, metrics, and early stopping callback
    optimizer = kr.optimizers.SGD(learning_rate=1e-2, momentum=0.9)
    loss = kr.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = kr.metrics.SparseCategoricalAccuracy()
    early_stopping = kr.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True)
    
    # Build and compile
    model.build(input_shape=(None, 784))
    model.compile(
        optimizer=optimizer,
        loss=loss, 
        metrics=[metrics]
    )

    # Train model
    history = model.fit(
        train_loader, 
        epochs=max_epochs,
        steps_per_epoch=np.floor(len(train_dataset) / batch_size), 
        verbose=0,
        validation_data=test_loader,
        validation_steps=1,
        callbacks=[early_stopping],
    )
    
    # Return history and final test accuracy
    test_acc = model.evaluate(test_dataset.batch(1000), verbose=0)
    return history, test_acc[1] # .evaluate returns [loss, accuracy]


# Iterating over all activation functions:

# In[17]:


# Recall shuffle, batch, repeat pattern to create epochs
BATCH_SIZE = 256
NUM_EPOCHS = 40

train_loader = train_dataset.shuffle(buffer_size=1000)
train_loader = train_loader.batch(batch_size=BATCH_SIZE, drop_remainder=True)
train_loader = train_loader.prefetch(buffer_size=BATCH_SIZE)
train_loader = train_loader.repeat(NUM_EPOCHS)

test_loader = test_dataset.shuffle(buffer_size=4096)
test_loader = test_loader.batch(2048)

results = {}
trained_models = {}
model_history = {}
_, ax = plt.subplots(2, 2, figsize=(12, 10))

for activation in activation_by_name.keys():
    model = base_network(activation)
    history, test_acc = train_model(
        model, 
        train_loader=train_loader,
        test_loader=test_loader,
        batch_size=BATCH_SIZE,
        max_epochs=NUM_EPOCHS,
    )
    results[activation] = test_acc
    trained_models[activation] = model
    model_history[activation] = history
    
    # Plotting train loss
    n = len(history.history['loss'])
    ax[0, 0].set_title("Train loss")
    ax[0, 0].plot(range(n), history.history['loss'], label=activation)
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    ax[1, 0].set_title("Valid loss")
    ax[1, 0].plot(range(n), history.history['val_loss'], label=activation)
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Plotting accuracy
    ax[0, 1].set_title("Train accuracy")
    ax[0, 1].plot(range(n), history.history['sparse_categorical_accuracy'], label=activation)
    ax[0, 1].legend()
    ax[0, 1].grid(True)
    ax[1, 1].set_title("Valid accuracy")
    ax[1, 1].plot(range(n), history.history['val_sparse_categorical_accuracy'], label=activation)
    ax[1, 1].legend()
    ax[1, 1].grid(True)


# Repeating the same experiment but not plotting sigmoid:

# In[18]:


_, ax = plt.subplots(2, 2, figsize=(12, 10))

for activation in activation_by_name.keys():
    history = model_history[activation]
    if activation != "sigmoid":

        # Plotting train loss
        n = len(history.history['loss'])
        ax[0, 0].set_title("Train loss")
        ax[0, 0].plot(range(n), history.history['loss'], label=activation)
        ax[0, 0].legend()
        ax[0, 0].grid(True)
        ax[1, 0].set_title("Valid loss")
        ax[1, 0].plot(range(n), history.history['val_loss'], label=activation)
        ax[1, 0].legend()
        ax[1, 0].grid(True)

        # Plotting accuracy
        ax[0, 1].set_title("Train accuracy")
        ax[0, 1].plot(range(n), history.history['sparse_categorical_accuracy'], label=activation)
        ax[0, 1].legend()
        ax[0, 1].grid(True)
        ax[1, 1].set_title("Valid accuracy")
        ax[1, 1].plot(range(n), history.history['val_sparse_categorical_accuracy'], label=activation)
        ax[1, 1].legend()
        ax[1, 1].grid(True)


# The final test accuracies are shown in the following table (these are from the best weights since we set `restore_best_weights` to `True` in our early stopping callback):

# In[19]:


import pandas as pd
pd.DataFrame({k: [results[k]] for k in results.keys()})


# Not surprisingly, the model using the sigmoid activation function has relatively bad performance. This is because of the low magnitudes of the gradients on layers near the input. All the other activation functions gain 
# similar performance.
# 
# To have a more accurate conclusion, we would have to train the models for multiple seeds and look at the averages.
# However, the "optimal" activation function also depends on many other factors (hidden sizes, number of layers, type of layers, task, dataset, optimizer, learning rate, etc.) so that a thorough grid search would not be useful in our case.
# 
# In the literature, activation functions that have shown to work well with deep networks are all types of ReLU functions we experiment with here, with small gains for specific activation functions in specific networks.

# ### Distribution of activation values

# After we have trained the models, we can look at the actual activation values that find inside the model. For instance, how many neurons are set to zero in ReLU? Where do we find most values in Tanh?
# To answer these questions, we can write a simple function which takes a trained model, applies it to a batch of images, and plots the histogram of the activations inside the network over the input batch. Note that we only look at the activations in the hidden and input layers excluding the logits layer (which is a linear layer).

# In[20]:


def plot_activations_distribution(model, activation, color="C0"):
    """Plot activation density for output of each layer for one forward pass."""

    # We need to manually loop through the layers to save all activations
    images, _ = fixed_batch
    activations = {0: images}
    for i, layer in enumerate(model.layers[:-1]):
        activations[i + 1] = layer(activations[i])

    # Plotting
    columns = 4
    rows = math.floor(len(activations) / columns)
    fig, ax = plt.subplots(rows, columns, figsize=(columns*2.7, rows*2.5))
    fig_index = 0
    
    for key in list(activations.keys())[:-1]: # key := layer_index
        key_ax = ax[divmod(fig_index, columns)]
        sns.histplot(data=activations[key].numpy().reshape(-1), bins=50, stat='density', ax=key_ax, color=color, kde=True)
        key_ax.set_title(f"({key}): {model.layers[key].name}")
        fig_index += 1
        
    fig.suptitle(f"Activation distribution for activation function {activation}", fontsize=12)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    return fig, ax


# Plotting activations distribution of trained models:

# In[21]:


for i, activation in enumerate(activation_by_name.keys()):
    plot_activations_distribution(trained_models[activation], activation, color=f"C{i}")


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
# To get a better understanding of how much of a problem this is, and when we need to be careful, we will measure how many dead neurons different networks have. For this, we implement a function which runs the network on the whole training set and records whether a neuron's activation magnitude is "small" (`<1e-8` by default) for majority of the data points (`0.99` by default):

# In[22]:


def fraction_dead_neurons(model, threshold=0.99, eps=1e-8):
    """Fraction of neurons with small activations on one batch."""

    # Initialize counter for each layer
    count_zero_actn = {}
    for i, layer in enumerate(model.layers[:-1]):
        if isinstance(layer, kr.layers.Dense):
            layer_width = layer.get_weights()[1].shape
            count_zero_actn[i + 1] = tf.zeros(layer_width)
    
    # Count zero activations during forward pass of one batch
    x = fixed_batch[0]
    for i, layer in enumerate(model.layers[:-1]):
        x = layer(x)
        if isinstance(layer, ActivationFunction):
            count_zero_actn[i] += tf.reduce_sum(tf.cast(tf.abs(x) < eps, tf.float32), axis=0)

    # Print fraction of dead neurons in each layer.
    # Dead = fraction of "zero" activations (over train set) exceeds threshold.
    print("Number of dead neurons:")
    for key in count_zero_actn.keys():
        num_dead = tf.reduce_sum(tf.cast((count_zero_actn[key] / len(x)) >= threshold, tf.float32))
        total = count_zero_actn[key].shape[0]
        print(f"  {f'({key})':>4} {model.layers[key].__class__.__name__}: {int(num_dead):3}/{total:3} {f'({(100.0 * num_dead / total):.2f}%)':>8}")


# **Zero network.** Testing with network with all neurons having zero output:

# In[23]:


def zeros_init(net):
    for layer in net.layers:
        if isinstance(layer, kr.layers.Dense):
            layer.set_weights([tf.zeros(w.shape) for w in layer.get_weights()])

# Initialize with zero weights
net_zero = base_network(activation='relu')
net_zero.build(input_shape=(None, 784))
zeros_init(net_zero)

# Should sum to zero
tf.print(sum([tf.reduce_sum(v) for v in net_zero.variables]))


# In[24]:


fraction_dead_neurons(net_zero) # Should be all dead


# **Untrained network.** Let's measure the number of dead neurons for an untrained network:

# In[25]:


net_relu = base_network(activation='relu', hidden_sizes=[512, 256, 256, 128])
net_relu.build(input_shape=(None, 784))
fraction_dead_neurons(net_relu)


# We see that only a minor amount of neurons are dead, but that they increase with the depth of the layer.
# In the long term, this is not a problem for the small number of dead neurons we have as the input to the neuron's layer is changed due to updates to the weights of lower layers. Such weights can be updated from gradients flowing into lower layers through alternate paths (i.e. across neurons in the same layer that are alive). Therefore, dead neurons can potentially become alive or active again. 

# **Trained network.** How does this look like for a trained network with the same initialization?

# In[26]:


trained_relu = trained_models['relu']
trained_relu.build(input_shape=(None, 784))
fraction_dead_neurons(trained_relu)


# The number of dead neurons indeed decreased in the later layers. However, it should be noted that dead neurons are especially problematic in the input layer since there are no lower layers whose weights can be updated to shift the activations to be predominantly positive. Still, the input data has usually a sufficiently high standard deviation to reduce the risk of dead neurons.

# **Untrained deep network.** Finally, we check how the number of dead neurons behaves with increasing layer depth. For instance, let’s take the following 10-layer neural network:

# In[27]:


net_relu = base_network(activation='relu', hidden_sizes=[256, 256, 256, 256, 256, 128, 128, 128, 128, 128])
net_relu.build(input_shape=(None, 784))
fraction_dead_neurons(net_relu)


# The number of dead neurons is significantly higher than before which harms the gradient flow especially in the first iterations. For instance, more than 25% of the neurons in the pre-last layer are dead which creates a considerable bottleneck. Hence, it is advisible to use other nonlinearities like Swish for very deep networks.

# In[28]:


net_swish = base_network(activation='swish', hidden_sizes=[256, 256, 256, 256, 256, 128, 128, 128, 128, 128])
net_swish.build(input_shape=(None, 784))
fraction_dead_neurons(net_swish)


# ## Conclusion
# 
# In this notebook, we have reviewed a set of six activation functions (sigmoid, tanh, ReLU, LeakyReLU, ELU, and Swish) in neural networks, and discussed how they influence the gradient distribution across layers. Sigmoid tends to fail deep neural networks as the highest gradient it provides is 0.25 leading to vanishing gradients in early layers. All ReLU-based activation functions have shown to perform well, and besides the original ReLU, do not have the issue of dead neurons. When implementing your own neural network, it is recommended to start with a ReLU-based network and select the specific activation function based on the properties of the network.

# In[ ]:




