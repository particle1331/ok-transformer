#!/usr/bin/env python
# coding: utf-8

# # Initialization and Optimization

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

# In this notebook, we will review techniques for optimization and initialization of neural networks. When increasing the depth of neural networks, there are various challenges we face. Most importantly, we need to have a stable gradient flow through the network, as otherwise, we might encounter vanishing or exploding gradients. This is why we will take a closer look at the following concepts: **initialization** and **optimization**.
# 
# In the first half of the notebook, we will review different initialization techniques, and go step by step from the simplest initialization to methods that are nowadays used in very deep networks. In the second half, we focus on optimization comparing the optimizers SGD, SGD with Momentum, and Adam.

# ```{margin}
# ⚠️ **Attribution:** This notebook builds on [Tutorial 4: Optimization and Initialization](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html). The original tutorial is written in PyTorch and is part of a lecture series on Deep Learning at the University of Amsterdam. The original tutorials are released under [MIT License](https://github.com/phlippe/uvadlc_notebooks/blob/master/LICENSE.md).
# ```

# In[1]:


import tensorflow as tf
import tensorflow.keras as kr
import tensorflow_datasets as tfds

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from tqdm import tqdm

from inefficient_networks import utils
from inefficient_networks.config import config 

config.set_matplotlib()
config.set_tensorflow_seeds(42)
config.set_ignore_warnings()
print(config.list_tensorflow_devices())
print(tf.__version__)


# ## Preliminaries

# Throughout this notebook, we will use a deep fully connected network, similar to our previous tutorial. We will also again apply the network to FashionMNIST, so you can relate to the results in the previous notebook. We start by loading the FashionMNIST dataset:
# 

# In[2]:


def transform_image(image, sample_mean, sample_std):
    """Flatten image standardized with respect to itself."""
    
    image = tf.cast(tf.reshape(image, (-1,)), tf.float32)
    return (image - sample_mean) / sample_std


# Load training data
FMNIST, FMNIST_info = tfds.load(
    'fashion_mnist', 
    data_dir=config.DATASET_DIR, 
    with_info=True, 
    shuffle_files=False
)
train_ds, test_ds = FMNIST['train'], FMNIST['test']

# Get pixel-wise sample statistics
images = tf.reshape(next(iter(train_ds.batch(60000)))['image'], (-1, 784))
images = tf.cast(images, tf.float32)
sample_mean = tf.reduce_mean(images, axis=0)
sample_std = tf.math.reduce_std(images, axis=0)

# Preprocess input data
train_ds = train_ds.map(lambda x: (transform_image(x['image'], sample_mean, sample_std), x['label']))
test_ds = test_ds.map(lambda x: (transform_image(x['image'], sample_mean, sample_std), x['label']))

# For all our analysis, we fix one batch
fixed_batch = next(iter(train_ds.batch(batch_size=4096)))


# The normalization is now designed to give us an expected mean of `0` and a standard deviation of `1` for each pixel across the whole sample. This shouldn't affect training performance since the network can learn to undo this preprocessing. 
# One difficulty is that if we ever deploy the model, we will have to worry about storing and tracking batch statistics for every training sample that we use to train our models. However, it will be particularly relevant for the discussion about initialization that we will look at below, and hence we perform it here. 

# In[3]:


fig, ax = plt.subplots(1, 1, figsize=(5, 2))
sns.histplot(fixed_batch[0].numpy().reshape(-1), binwidth=0.1, stat='density', element='poly');
ax.set_xlim([-3, 5])
ax.set_xlabel("Pixel intensity");


# Next, we define a function for instantiating a neural network. Here `activation` can be any TensorFlow callable function. In particular, functions in the `activations` library of Keras. The initializer in `kernel_initializer` is any valid layer weight initializer for Keras forward layers. In general, any callable that takes `shape` (of the weight tensor to initialize) and `dtype` (of initialized value) can be passed to this argument. 

# In[4]:


def base_network(
    activation, 
    kernel_initializer,
    num_classes=10, 
    hidden_sizes=(512, 256, 256, 128)
):
    """Return a fully-connected network with given activation and layer widths."""

    # Add hidden layers with activation
    model = kr.Sequential()
    for j in range(len(hidden_sizes)):
        model.add(
            kr.layers.Dense(
                hidden_sizes[j],
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer='zeros'
            )
        )

    # Add linear logits layer
    model.add(
        kr.layers.Dense(
            units=num_classes,
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros'
        )
    )
    
    return model


# For the activation functions, we make use of Keras' `activations` library instead of implementing ourselves. We also use `tf.identity` are our identity activation function. Although this activation function would significantly limit the network’s modeling capabilities, we will use it in the first steps of our discussion about initialization.

# In[5]:


activations_by_name = {
    "tanh": kr.activations.tanh,
    "relu": kr.activations.relu,
    "identity": tf.identity
}


# Finally, we define a few plotting functions that we will use for our discussions. These functions help us to visualize 1) the gradients that the parameters at different layers receive, and 2) the activations, or the output, of the linear layers. The detailed code is not important, but feel free to take a closer look if interested.

# In[6]:


def plot_distributions(dist: dict, color="C0", stat="count", xlabel=None, use_kde=True, xlim=None):
    """Helper function for plotting histograms from numpy arrays."""
    
    cols = len(dist)
    fig, ax = plt.subplots(1, cols, figsize=(cols*3, 3.0), dpi=80)
    fig_index = 0

    for key in sorted(dist.keys()):
        # Plot distribution
        ax_key = ax[fig_index % cols]
        sns.histplot(
            dist[key], ax=ax_key, color=color, bins=50, stat=stat, 
            kde=use_kde and ((dist[key].max() - dist[key].min()) > 1e-8) # plot KDE only if nonzero variance
        ) 
        
        # Formatting
        ax_key.set_xlabel(xlabel)
        ax_key.set_xlim(xlim)
        ax_key.set_title(str(key))
        fig_index += 1
    
    fig.subplots_adjust(wspace=0.4)
    return fig


def plot_gradient_distribution(model, color="C0", print_variance=False, xlim=None):
    """Plot gradient histogram for the kernel of each layer for one forward pass."""
    
    # Get fixed sample
    images, labels = fixed_batch

    # Pass the batch through the network, and calculate the gradients for the weights
    loss_fn = kr.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape(persistent=True) as tape:
        preds = model(images)
        loss = loss_fn(labels, preds)
    
    # Exclude the bias to reduce the number of plots
    grads_dict = {}
    for layer_index, layer in enumerate(model.layers):
        grads = tape.gradient(loss, layer.variables[0])
        grads_dict[(layer_index, f"{layer.name.split('_')[0]}")] = grads.numpy().reshape(-1)

    # Plotting
    fig = plot_distributions(grads_dict, color=color, stat='count', xlabel="Gradient", xlim=xlim)
    fig.suptitle(f"Gradient magnitude distribution (weights)", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Print variances of each distribution
    if print_variance:
        for key in sorted(grads_dict.keys()):
            print(f"{str(key):<12}  σ²_grad = {np.var(grads_dict[key]):.5e}")

    return grads_dict


def plot_activations_distribution(model, color="C0", print_variance=False, xlim=None):
    """Plot activation density for output of each layer for one forward pass."""

    # Get fixed data
    images, _ = fixed_batch

    # Store activations per layer
    activations = {}
    x = images
    for layer_index, layer in enumerate(model.layers):
        x = layer(x)
        activation_name = str(layer.activation).strip().split()[1] # weird hack
        activations[(layer_index, f"{activation_name}")] = x.numpy().reshape(-1)

    # Plotting
    fig = plot_distributions(activations, color=color, stat='density', xlabel="Activation", xlim=xlim)
    fig.suptitle("Activation values distribution", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Print variances of each distribution
    if print_variance:
        for key in sorted(activations.keys()):
            print(f"{str(key):<15}  σ²_act = {np.var(activations[key]):.5e}")

    return activations


# **Remark.** Observe that neurons in the hidden layers are identically distributed during initialization by symmetry. This allows us to combine all values in the `(B, m)` matrix into a histogram of `B*m` samples whose distribution should reflect the distribution of each individual output neuron. But this doesn't hold for neurons in the input and output layers which have different distributions. Thus, histogram plots for the input and output layers should be interpreted as an aggregated distribution which will generally differ from individual distributions of each neuron.

# ## Initialization

# Training deep neural networks is essentially an optimization problem (in very high dimensions) with the network weights as parameters and the loss as objective. Thus, we have to choose initial values for the weights.
# When initializing a neural network, there are a few properties we would like to have. 
# 
# First, the variance of the input should be propagated through the model to the last layer, so that we have a similar standard deviation for the output neurons. If the variance would vanish the deeper we go in our model, it becomes much harder to optimize the model as the input to the next layer is basically a single constant value. Similarly, if the variance increases, it is likely to explode (i.e. head to infinity) the deeper we design our model. The second property we look out for in initialization techniques is a gradient distribution with equal variance across layers. If the first layer receives much smaller gradients than the last layer, we will have difficulties in choosing an appropriate learning rate.
# 
# As a starting point for finding a good method, we will analyze different initialization based on our linear neural network with no activation function (i.e. an identity). We do this because initializations depend on the specific activation function used in the network, and we can adjust the initialization schemes later on for our specific choice.
# 
# 

# In[7]:


model = base_network(activation=None, kernel_initializer='glorot_normal') # = linear transformation
model.build(input_shape=(None, 784))
model.summary()


# To easily visualize various initialization schemes, we define the following helper function. Recall that the bias weights are initialized to zero in all subsequent discussion.

# In[8]:


def visualize_initialization(
    activation,
    initialization,
    plot_grad=True, 
    plot_act=True, 
    xlim_grad=None, 
    xlim_act=None
):
    """Helper function for visualizing different initialization schemes."""
    
    model = base_network(activation, initialization)
    model.build(input_shape=(None, 784))
    gradients = {}
    activations = {}

    if plot_grad:
        gradients = plot_gradient_distribution(
            model, color="C1",
            print_variance=True,
            xlim=xlim_grad,
        )
    if plot_act:
        activations = plot_activations_distribution(
            model, color="C2", 
            print_variance=True,
            xlim=xlim_act,
        )

    return {
        'gradients': gradients,
        'activations': activations
    }


# ### Constant initialization

# The first initialization we can consider is to initialize all weights with the same constant value. Using a large constant will make the network have exploding activations since neurons accumulate the input vector into a weighted sum. What happens if we set all weights to a value slightly larger or smaller than 0? To find out, we can implement a function for setting all parameters below and visualize the gradients. However, thinking a bit deeper, we see that setting all weights to some constant is not a good idea as the network will have trouble breaking symmetry between neurons.

# In[9]:


result = visualize_initialization(
    activation=None, # Identity
    initialization=kr.initializers.Constant(value=0.005),
)


# As we can see, only the first and the last layer have diverse gradient distributions each intermediate hidden layers have the same gradient for all weights (note that this value is unequal to 0, but very close to it). Due to symmetry, all intermediate neurons belonging to the same layer will be equivalent, and therefore have the same gradient update so that `σ²_grad=0.0`, essentially reducing the effective number of parameters to 1 for these layers. 
# 
# The only sources of assymetry are the inputs and outputs, which explains the nonzero variance in the gradients of the first and last layers. For example, pixels in the edge of input images have different distributions with pixels on the center.

# ### Constant variance
# 

# From the experiment above, we have seen that a constant value is not working. So to break symmetry, how about we initialize the parameters by randomly sampling from a distribution like a Gaussian? The most intuitive way would be to choose one variance that is used for all layers in the network. Let’s implement it below, and visualize the activation distribution across layers.

# In[10]:


result = visualize_initialization(
    activation=None,
    initialization=kr.initializers.RandomNormal(0, stddev=0.01),
    xlim_grad=[-0.1, 0.1],
    xlim_act=[-2, 2],
)


# The variance of the activation becomes smaller and smaller across layers, and almost vanishes in the last layer. Alternatively, we could use a higher standard deviation:
# 

# In[11]:


result = visualize_initialization(
    activation=None, 
    initialization=kr.initializers.RandomNormal(stddev=0.8),
    xlim_grad=[-30000, 30000],
    xlim_act=[-3500000, 3500000]
)


# With a higher standard deviation, the activations are likely to explode. We also get vanishing gradients. You can play around with the specific standard deviation values, but it will be hard to find one that gives us a good activation distribution across layers and is very specific to our model. If we would change the hidden sizes or number of layers, you would have to search all over again, which is neither efficient nor recommended!

# ### How to find appropriate initialization values
# 
# Suppose we want to design an initialization for the linear layer which computes $\mathbf y= \mathbf x \boldsymbol W + \boldsymbol b$ with $\mathbf y\in\mathbb{R}^{h_{\mathbf y}}$, $\mathbf x\in\mathbb{R}^{h_{\mathbf x}}$. From our experiments above, we saw that we need to optimally sample weights to ensure healthy distribution of activation values. For this, we state two requirements: 
# 
# 1. The mean of the activations should be zero.
# 2. The variance of the activations should stay the same across every layer. 
# 
# Note that the activation neurons in a layer are identically distributed as a consequence of symmetry in the MLP network and the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) (CLT), so we can write the variance as $\sigma_{\mathbf y}^{2} = \mathbb V(y_j).$ It follows that $\boldsymbol b = \mathbf 0$ since the bias is constant across different inputs. Next, it makes sense to set the mean of the weights to zero for the sake of symmetry. This also means that we only have to calculate the variance which can be done as follows:
# 
# $$\begin{split}
#     \sigma_{\mathbf y}^{2} = \mathbb V(y_j) & = \mathbb V\left(\sum_{i} x_{i}w_{ij}\right)\\
#     & = \sum_{i}\ \mathbb V(x_{i}w_{ij})\\
#     & = \sum_{i}\ \mathbb V(x_{i}) \,\mathbb V(w_{ij})\\
#     & = h_{\mathbf x}\,\mathbb V(x_{i})\,\mathbb V(w_{ij}) = h_{\mathbf x}\, \sigma_{\mathbf x}^{2} \,\sigma_{\boldsymbol W}^2.
# \end{split}
# $$
# 
# Since the columns of $\boldsymbol W$ are independent of each other, the second line follows from the [variance of the sum of independent random variables](https://muchomas.lassp.cornell.edu/8.04/Lecs/lec_statistics/node14.html), while the third line follows from the [variance of a product of two independent random variables with zero mean](https://stats.stackexchange.com/questions/52646/variance-of-product-of-multiple-independent-random-variables). Note that $\mathbf x$ also has zero mean  by the inductive hypothesis. Finally, we get the last equality since each $x_i$ and $w_{ij}$ are identically distributed.
# 
# **Remark.** As a technical aside, note that while the layers have identical and distributions, and zero mean, the input layer does not. This isn't an issue, since all analysis can start with the neurons in the output of the input layer, since by CLT these have zero mean and identical distributions. Note that CLT only works since each neuron in the input layer has approximately zero mean by our preprocessing methodology. 

# In[12]:


model = base_network(activation=None, kernel_initializer='glorot_normal') # = linear transformation
prep_inputs, _ = fixed_batch
activations = {0: prep_inputs}
x = prep_inputs
for i, layer in enumerate(model.layers):
    x = layer(x)
    activations[i + 1] = x

# Plot activations distribution for different pixels
cols = len(activations)
fig, ax = plt.subplots(1, cols, figsize=(cols*3, 3.0))

for key in sorted(activations.keys()):
    # For input layer, we choose pixels on different local regions of the image.
    # For hidden layers, all pixels are iid, so we can choose adjacent neurons.
    if key == 0:
        p1, p2 = 100, 500 # pixel indices
    else:
        p1, p2 = 0, 1
        
    ax_key = ax[key % cols]
    sns.histplot(activations[key].numpy()[:, p1], ax=ax_key, color=f"C0", stat='density', element='poly') 
    sns.histplot(activations[key].numpy()[:, p2], ax=ax_key, color=f"C1", stat='density', element='poly')
    ax_key.set_xlabel("Activation")
    ax_key.set_title(f"layer {key}")

fig.subplots_adjust(wspace=0.4)    
plt.suptitle("Activation distribution of two neurons in network layers.", fontsize=14, y=1.05);


# Going back to the derivation, it follows that to get $\sigma^2_{\mathbf y} = \sigma^2_{\mathbf x},$ we must have $\sigma_{\boldsymbol W}^2 = \frac{1}{h_{\mathbf x}}.$ That is, we should initialize the weight distribution with a variance equal to the inverse of the layer's input dimension or number of *fan-in* neurons. Let's implement this below and check whether we get better results:

# In[13]:


def xavier_fanin(shape, dtype=None):
    fan_in = shape[0]
    return kr.initializers.RandomNormal(0, stddev=1/np.sqrt(fan_in))(shape)

result = visualize_initialization(
    activation=None, 
    initialization=xavier_fanin,
    xlim_grad=[-0.3, 0.3],
    xlim_act=[-5, 5],
)


# As we expected, the variance stays indeed constant across layers. Note that our initialization does not restrict us to a normal distribution, but allows any other distribution with a mean of $0$ and variance of $\frac{1}{h_{\mathbf x}}.$ You often see that a uniform distribution is used for initialization. A small benefit of using a uniform instead of a normal distribution is that we can exclude the chance of initializing very large or small weights.

# In the above plot, we see that gradients slightly vanish nearer the inputs. Indeed, besides the variance of the activations, another variance we would like to stabilize is the one of the gradients.  This ensures a stable optimization for deep networks. From our work on backpropagation on MLPs, we know that
# $\frac{\partial \mathcal L}{\partial \mathbf x} = \frac{\partial \mathcal L}{\partial \mathbf y} \boldsymbol W^\top.$
# Hence
# $\sigma^2_{\boldsymbol W^\top} = \sigma^2_{\boldsymbol W} = \frac{1}{h_\mathbf y}.$
# As a compromise between both constraints, in [[Glorot and Bengio (2010)]](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi) the authors proposed to use the harmonic mean of both values. This leads us to the well-known **Xavier initialization**. For a normal distribution of initial weights, this looks like:
# 
# $$\boldsymbol W\sim \mathcal{N}\left(\mu = 0,\sigma^2=\frac{2}{h_{\mathbf x}+h_{\mathbf y}}\right).$$
# 
# If we use a uniform distribution, we initialize the weights with:
# 
# $$\boldsymbol W\sim U\left[-\frac{\sqrt{6}}{\sqrt{{h_{\mathbf x}+h_{\mathbf y}}}}, \frac{\sqrt{6}}{\sqrt{{h_{\mathbf x}+h_{\mathbf y}}}}\right].$$
# 
# Let's shortly implement it and validate its effectiveness:

# In[14]:


def xavier(shape, dtype=None):
    fan_in, fan_out = shape
    fan_avg = 0.5 * (fan_in + fan_out)
    init = kr.initializers.RandomNormal(0, stddev=np.sqrt(1/fan_avg))
    return init(shape)

result = visualize_initialization(
    activation=None,
    initialization=xavier,
)


# We see that the Xavier initialization makes the variance of gradients and activations consistent across layers (`σ²_act` around `2` and `σ²_grad` around `1e-3`). Note that the significantly higher variance for the output layer is due to the large difference of input and output dimension (128 vs 10), so that both `fan_in` and `fan_out` differs significantly from `fan_avg`.

# #### Xavier initialization of $\tanh$ networks

# In the discussions above, we assumed the activation function to be linear. So what happens if we add a non-linearity? In a tanh-based network, a common assumption is that for small values during the initial steps in training, the $\tanh$ works as a linear function such that we don’t have to adjust our calculation. We can check if that is the case for us as well.

# Observe that accumulation of activation values with large weights pushes $\tanh$ activations to $\pm 1$ which likewise will continue to the rest of the network's layers. This limits the expressivity of the network.

# In[15]:


visualize_initialization(
    activation='tanh',
    initialization=kr.initializers.RandomNormal(0, stddev=0.3),
    plot_grad=False
);


# For small fixed $\sigma,$ we get similar behavior with the identity network since for small input, $\tanh x \approx x.$ Thus, we get vanishing activations.

# In[16]:


visualize_initialization(
    activation='tanh',
    initialization=kr.initializers.RandomNormal(0, stddev=0.01),
    plot_grad=False,
    xlim_act=[-1, 1]
);


# Let's try to initialize with Xavier normalization. This should work fairly well since $\tanh$ is approximately linear between -1 and 1. Indeed, observe that we get healthier activation distribution compared to initializing the weights with constant variance above!

# In[17]:


result = visualize_initialization(
    activation='tanh',
    initialization=xavier,
)


# #### Kaiming initialization for ReLU networks

# But what about ReLU networks? Here, we cannot take the previous assumption of the non-linearity becoming linear for small values.  Suppose $\mathbf y = \mathbf x\boldsymbol W$ such that $\mathbf x$ is an output of a ReLU activated layer. So $\mathbf x$ is not any more zero-centered. But as long as the expectation of $\boldsymbol W$ is zero and $\boldsymbol b= \mathbf 0$, the expectation of the output is zero, so we still have mean zero for $\mathbf y.$ Here our goal is for preactivations to have constant variance and zero mean as this should result in controlled activations. For the latter requirement, the part where the calculation of the ReLU initialization differs from the identity is when determining:
# 
# $$
# \mathbb V(x_{i} w_{ij} ) = 
# \underbrace{\mathbb{E}[{x_{i}}^2]}_{\mathbb V(x_i)} \;
# \underbrace{\mathbb{E} [ {w_{ij}}^2 ]}_{\mathbb V(w_{ij})}
# -
# \mathbb{E}[ x_{i} ]^2 \; \underbrace{\mathbb{E}[w_{ij}]^2}_{= 0}
# = \mathbb{E}[{x_{i}}^2] \; \mathbb V(w_{ij}).
# $$
# 
# If we assume now that $\mathbf x$ is the output of a ReLU activation, we can calculate the expectation as follows. In the first equality, $\rho$ is the probability distribution of the values of the preactivation neuron $z_i$ that outputs $x_i.$ We can assume that $\rho$ is symmetric around zero by the inductive hypothesis. Thus
# 
# $$
# \begin{split}
# \mathbb{E}[{x_i}^2] 
# &= \int_{-\infty}^{\infty} \max(0, u)^2 \rho(u)\, du \\
# &= \int_0^{\infty} u^2 \rho(u)\, du = \frac{1}{2}\int_{-\infty}^{\infty} u^2 \rho(u)\, du = \frac{1}{2}\mathbb V(z_i).
# \end{split}$$
# 
# 
# It follows that $\sigma^2_{\mathbf y} = \frac{1}{2}\sum_{j} \sigma^2_{\boldsymbol W}\, \sigma^2_{ {\mathbf z}}= \frac{1}{2}\, h_\mathbf{x}\, \sigma^2_{\boldsymbol W}\, \sigma^2_{ {\mathbf z}}.$ So our desired weight variance becomes $\sigma^2_{\boldsymbol W} = \frac{2}{h_{\mathbf x}}$ which is the well-known **Kaiming initialization** [[He, K. et al. (2015)]](https://arxiv.org/pdf/1502.01852.pdf). Note that the Kaiming initialization does not use the harmonic mean between input and output size. In the paper, the authors argue that using $h_{\mathbf x}$ or $h_{\mathbf y}$ both lead to stable gradients throughout the network, and only depend on the overall input and output size of the network. Hence, we can use here only the input $h_{\mathbf x}.$

# In[18]:


def kaiming(shape, dtype=None):
    fan_in, fan_out = shape
    init = kr.initializers.RandomNormal(0, stddev=np.sqrt(2/fan_in))
    return init(shape)

result = visualize_initialization(
    activation='relu',
    initialization=kaiming,
    plot_grad=False,
)


# In contrast, having no factor of 2 results in vanishing activation values. The activation variance becomes  unstable across layers. 

# In[19]:


visualize_initialization(
    activation='relu',
    initialization=xavier,
    plot_grad=False,
);


# We can conclude that the Kaiming initialization indeed works well for ReLU-based networks. Note that for other activations we have to slightly adjust the factor in the variance. For instance, for LeakyReLU half of the values are not set to zero anymore, and calculating a similar integral as above results in a factor of $\frac{2}{1 + \alpha^2}$ instead of $2$ for the ReLU.
# 
# ```{note}
# To initialize with different scale factors, Keras implements the `VarianceScaling` initializer which samples weights from a normal distribution with mean zero and standard deviation `stddev = sqrt(scale / n)` where `n` depends on the `mode` with `'fan_in'`, `'fan_out'`, and `'fan_avg'` as possible values. In particular, Kaiming corresponds to `scale=2` and `mode='fan_in'`. Xavier corresponds to `scale=1` and `mode='fan_avg'`. Note that same with other normal initializers, Keras truncates the sample space to prevent initializing too large weights. 
# ```

# ### Understanding activation and gradient flow

# **Understanding activations.** Suppose we index layers and weights as in {numref}`neuralnet-layers` with $\mathbf x_0$ as input data. It follows that $\sigma_{\mathbf x_{t+1}}^{2} = \sigma_{\mathbf x_{t}}^{2}\,h_{\mathbf x_{t}}\, \sigma_{\boldsymbol W_t}^2$ for $t \geq 1.$ Thus, applying the formula recursively, we get
# 
# $$\sigma_{\mathbf x_{t}}^{2} =  \sigma_{\mathbf x_{1}}^{2} \left(\prod_{k=1}^{t-1} h_{\mathbf{x}_k}\,\sigma_{\boldsymbol W_k}^2\right)$$ 
# 
# for $t \geq 1.$ This formula explains why activations and gradients blow up as we go deeper into the layers for a network initialized with sufficiently large constant variance for the weights, and vanishes with depth for sufficiently small constant variance. For example, $\sigma_{\boldsymbol W_k}^2 = \frac{1}{h_{\mathbf x_{k}}}$ in Xavier initialization, so that $\sigma_{\mathbf x_{t}}^{2} = \sigma_{\mathbf x_{1}}^{2}.$ Note that we have the same equation for gradients but in the reverse direction (starting from the logits layer) which motivates fan-out Xavier initialization.

# ```{figure} ../../img/neuralnet-layers.png
# ---
# width: 30em
# name: neuralnet-layers
# ---
# Schematic diagram of a feedforward neural network. 
# ```

# Note that this equation is only an approximation for actual networks since neurons are only approximately independent and identically distributed, and also only have approximately zero mean. But we will test whether our computations are consistent at least in order of magnitude. Recall the identity network initialized with constant variance $\sigma = 0.8$ had exploding activations while for $\sigma=0.01$ it had vanishing activations.

# In[20]:


def test_activation_formula(std):
    model = base_network(
        activation=None, 
        kernel_initializer=kr.initializers.RandomNormal(stddev=std)
    )
    activations = {0: fixed_batch[0]}
    for i, layer in enumerate(model.layers):
        activations[i + 1] = layer(activations[i])

    for i in range(1, 5):
        v1 = activations[i].numpy().std() ** 2
        v2 = activations[i+1].numpy().std() ** 2
        print((activations[i].shape[1] * v1 * (std)**2, v2))
    

# Testing for exploding activations
test_activation_formula(std=0.8)


# In[21]:


# Testing for vanishing activations
test_activation_formula(std=0.01)


# ```{margin}
# The formula is similar for general nonlinear activations, but involves the derivative of the activation multiplied (or broadcasted) to the respective weight matrix. Hence, we expect a similar behavior.
# ```
# 
# **Understanding gradient flow.** For the input layer $\mathbf x_1 = \mathbf x_0 \boldsymbol W_0,$ we get $\frac{\partial \mathcal L}{\partial \boldsymbol W_0} = {\mathbf x_0^\top} \frac{\partial \mathcal L}{\partial \mathbf x_1}$ by backpropagating from $\mathbf x_1$ to $\boldsymbol W_0.$ Similarly, we can backpropagate from $\mathbf x_2$ to $\mathbf x_1$ in the next layer $\mathbf x_2 = \mathbf x_1 \boldsymbol W_1$ to get
# $\frac{\partial \mathcal L}{\partial \mathbf x_1} = \frac{\partial \mathcal L}{\partial \mathbf x_2} \boldsymbol W_1^\top.$
# Continuing this process, we get the weight gradient of the input layer in terms of the weight gradients of the logits layer $\mathbf x_5$ (which we have easy access to):
# 
# $$\frac{\partial \mathcal L}{\partial \boldsymbol W_0} = {\mathbf x_0^\top} \frac{\partial \mathcal L}{\partial \mathbf x_5} (\boldsymbol W_1\boldsymbol W_2 \boldsymbol W_3 \boldsymbol W_4)^\top$$
# 
# Shifting the starting point to get the gradient of any intermediate layer of the network:
# 
# $$\frac{\partial \mathcal L}{\partial \boldsymbol W_t} = {\mathbf x_t^\top} \frac{\partial \mathcal L}{\partial \mathbf x_d}\left( \boldsymbol W_{t+1} \ldots \boldsymbol W_{d-1} \right)^\top$$ 
# 
# where $0 \leq t \leq d-1.$ Notice the stack of weight matrices &mdash; this product can explode or vanish depending on the magnitude of the weights. From this formula, we can see why there is vanishing gradients with exploding activations for the identity network initialized with $\sigma_{\mathbf W_{k}} = 0.8.$ Since the activations explode, shallower layers have lower activations, hence get lower weight gradients because of the factor ${\mathbf x_t^\top}.$ Moreover, the entries of the weights product approaches zero. To further check for correctness, we implement the formula for $\frac{\partial \mathcal L}{\partial \boldsymbol W_t}$ in code as follows:

# In[22]:


def weights_gradient_formula(t, model):
    """Formula for computing gradient as a product of weights."""
    
    # Compute activations; here we are careful about indexing
    inputs, targets = fixed_batch
    acts_dict = {}
    x = inputs
    acts_dict[0] = x.numpy() # x0 = input
    for layer_index, layer in enumerate(model.layers):
        x = layer(x)
        acts_dict[layer_index + 1] = x.numpy() # x1 = x0 W0, etc.

    # Compute gradients of logits
    loss_fn = kr.losses.SparseCategoricalCrossentropy(from_logits=True)
    with tf.GradientTape(persistent=True) as tape:
        preds = model(inputs)
        loss = loss_fn(targets, preds)
    preds_grad = tape.gradient(loss, preds).numpy()

    # Compute return value
    X = acts_dict[t]
    weights = [h.get_weights()[0] for h in model.layers]
    if t > len(weights) - 1:
        raise IndexError
    else:
        import functools
        return functools.reduce(
            lambda x, y: x @ y.T, 
            weights[len(weights) - 1: t: -1], # Right multiply weights starting from index 0
            X.T @ preds_grad # Initial factor
        )


def weights_gradient_autodiff(model):
    """Compute gradients using autodifferentiation."""

    images, labels = fixed_batch
    loss_fn = kr.losses.SparseCategoricalCrossentropy(from_logits=True)
    grads_dict = {}
    with tf.GradientTape(persistent=True) as tape:
        loss = loss_fn(labels, model(images))

    for layer_index, layer in enumerate(model.layers):
        grads = tape.gradient(loss, layer.variables[0])
        grads_dict[layer_index] = grads.numpy()

    return grads_dict


# Testing if the above formula works:

# In[23]:


# Fix model
model = base_network(
    activation=None,
    kernel_initializer=xavier,
    hidden_sizes=[512, 256, 256, 128]
)
model.build(input_shape=(None, 784))

# Compute errors for each layer
errors = []
grads_dict = weights_gradient_autodiff(model)
for t in grads_dict.keys():
    W_grad = weights_gradient_formula(t, model)
    e = np.abs(W_grad - grads_dict[t]).max() / np.abs(grads_dict[t]).mean()
    errors.append(e)

plt.plot([str(k) for k in grads_dict.keys()], errors) # force int xticks
plt.xlabel('layer no.')
plt.ylabel('max abs. relative error');


# The deeper we get, the larger the error. So the small differences are probably roundoff errors or some implementation detail in TensorFlow regarding autodifferentiation. 

# ## Optimization

# Besides initialization, selecting a suitable optimization algorithm can be an important choice for deep neural networks. First, we need to understand what an optimizer actually does. The optimizer is responsible to update the network's parameters given the gradients. Hence, we effectively implement a function ${\boldsymbol w}^{t} = f({\boldsymbol w}^{t-1}, {\boldsymbol g}^{t}, ...)$ with $\boldsymbol w$ being the parameters, and ${\boldsymbol g}^{t} = \nabla_{{\boldsymbol w}^{(t-1)}} \mathcal{L}^{(t)}$ the gradients at time step $t$. A common, additional parameter to this function is the learning rate, here denoted by $\eta$. Usually, the learning rate can be seen as the "step size" of the update. A higher learning rate means that we change the weights more in the direction of the gradients, a smaller means we take shorter steps. 

# ### Optimization algorithms
# 
# As most optimizers only differ in the implementation of $f$, we can define a template for an optimizer below. We take as input the parameters of a model and a learning rate. The `step()` function tells the optimizer to update all weights based on their gradients.

# In[24]:


class OptimizerTemplate:
    def __init__(self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, grads):
        for w, g in zip(self.model.trainable_variables, grads):
            self.update_param(w, g)
    
    def update_param(self, weight, grad):
        raise NotImplementedError


# #### SGD

# The first optimizer we are going to implement is the standard Stochastic Gradient Descent (SGD). SGD updates the parameters using the following equation:
# 
# $$
# \begin{split}
#     {\boldsymbol w}^{(t)} & = {\boldsymbol w}^{(t-1)} - \eta\, {\boldsymbol g}^{(t)}.
# \end{split}
# $$
# 
# Let's implement this in the following class:

# In[25]:


class SGD(OptimizerTemplate):
    def __init__(self, model, learning_rate):
        super().__init__(model, learning_rate)

    def update_param(self, weight, grad):
        dw = -self.learning_rate * grad
        weight.assign_add(dw)


# #### SGD with Momentum

# SGD can be improved using the concept of **momentum** which replaces the gradient in the update by an exponential average of all past gradients including the current one:
# 
# $$
# \begin{split}
#     \boldsymbol{m}^{(t)} & = \beta_1 \boldsymbol{m}^{(t-1)} + (1 - \beta_1)\, {\boldsymbol g}^{(t)}\\
#     {\boldsymbol w}^{(t)} & = {\boldsymbol w}^{(t-1)} - \eta\, \boldsymbol{m}^{(t)}.\\
# \end{split}
# $$
# 
# Momentum help smooth out gradient updates. This can be helpful when dealing with oscillating updates. 

# In[26]:


class SGDMomentum(OptimizerTemplate):
    def __init__(self, model, learning_rate, momentum=0.0):
        super().__init__(model, learning_rate)
        self.beta = momentum
        self.m = {w.name: tf.zeros_like(w) for w in model.trainable_variables}

    def update_param(self, weight, grad):
        self.m[weight.name] = self.beta * self.m[weight.name] + (1 - self.beta) * grad
        dw = -self.learning_rate * self.m[weight.name]
        weight.assign_add(dw)


# #### Adam

# Finally, we arrive at Adam. Adam combines the idea of momentum with an adaptive learning rate, which is based on an exponential average of the squared gradients, i.e. the gradients norm. Furthermore, we add a bias correction for the momentum and adaptive learning rate for the first iterations:
# 
# $$
# \begin{split}
#     {\boldsymbol m}^{(t)} & = \beta_1 {\boldsymbol m}^{(t-1)} + (1 - \beta_1)\, {\boldsymbol g}^{(t)}\\
#     {\boldsymbol v}^{(t)} & = \beta_2 {\boldsymbol v}^{(t-1)} + (1 - \beta_2)\, ({\boldsymbol g}^{(t)})^2\\
#     \hat{{\boldsymbol m}}^{(t)} & = \frac{{\boldsymbol m}^{(t)}}{1-{\beta_1}^{t}},\; \hat{{\boldsymbol v}}^{(t)} = \frac{{\boldsymbol v}^{(t)}}{1-{\beta_2}^{t}}\\
#     {\boldsymbol w}^{(t)} & = {\boldsymbol w}^{(t-1)} - \frac{\eta}{\sqrt{\hat{{\boldsymbol v}}^{(t)}} + \epsilon} \hat{\boldsymbol m}^{(t)}\\
# \end{split}
# $$
# 
# Here $\epsilon$ is a small constant used to improve numerical stability for very small gradient norms. Remember that the adaptive learning rate does not replace the learning rate hyperparameter $\eta,$ but rather acts as an extra factor and ensures that the gradients of various parameters have a similar norm. 

# In[27]:


class Adam(OptimizerTemplate):
    def __init__(self, model, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # (t = 0)
        self.m = {w.name: tf.zeros_like(w) for w in model.trainable_variables}
        self.v = {w.name: tf.zeros_like(w) for w in model.trainable_variables}
        self.param_step = {w.name: 0 for w in model.trainable_variables} # time steps for each variable

    def update_param(self, weight, grad):
        # fetch prev. momentum and second momentum
        self.m[weight.name] = self.beta1 * self.m[weight.name] + (1 - self.beta1) * grad
        self.v[weight.name] = self.beta2 * self.v[weight.name] + (1 - self.beta2) * (grad ** 2)
        
        # bias correction
        self.param_step[weight.name] += 1
        t = self.param_step[weight.name]
        m_hat = self.m[weight.name] / (1 - self.beta1**t)
        v_hat = self.v[weight.name] / (1 - self.beta2**t)

        # update weights
        dw = -self.learning_rate * (m_hat / (tf.sqrt(v_hat) + self.eps))
        weight.assign_add(dw)


# ### Comparing optimizers on model training
# 
# After we have implemented three optimizers (SGD, SGD with momentum, and Adam), we can start to analyze and compare them. 
# First, we test them on how well they can optimize a neural network on the FashionMNIST dataset. We use again our linear network, this time with a ReLU activation and the Kaiming initialization, which we have found before to work well for ReLU-based networks. Note that the model is over-parameterized for this task, and we can achieve similar performance with a much smaller network (for example `100, 100, 100`). However, our main interest is in how well the optimizer can train *deep* neural networks, hence the over-parameterization.

# In[28]:


model = base_network(activation='relu', kernel_initializer='he_normal') # relu + kaiming.
model.build(input_shape=(None, 784))


# Let's define a training function.

# In[29]:


from sklearn import metrics

def train_model(model, optim, max_epochs=40, batch_size=256):
    # loss function
    loss_fn = kr.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Recall shuffle, batch, repeat pattern to create epochs
    train_loader = train_ds.shuffle(buffer_size=10000)
    train_loader = train_loader.batch(batch_size=batch_size, drop_remainder=True)
    train_loader = train_loader.repeat(max_epochs)
    train_loader = train_loader.prefetch(buffer_size=batch_size)    # Prepare next elements 
                                                                    # while current is preprocessed. 
                                                                    # Trades off latency with memory.

    valid_loader = test_ds.shuffle(buffer_size=4096)
    valid_loader = valid_loader.batch(2048)
    valid_loader = valid_loader.repeat()
    valid_iterator = iter(valid_loader)

    # training
    train_loss = []
    valid_loss = []
    valid_acc = []
    step = 0
    for x_train, y_train in tqdm(train_loader):
        with tf.GradientTape() as tape:
            loss = loss_fn(y_train, model(x_train))

        grads = tape.gradient(loss, model.trainable_variables)
        optim.step(grads)
        train_loss.append(loss)

        # compute valid. loss and valid. accuracy
        if step % batch_size == 0:
            x_valid, y_valid = next(valid_iterator)
            valid_loss.append(loss_fn(y_valid, model(x_valid)))
            valid_acc.append(
                metrics.accuracy_score(
                    y_valid, 
                    model.predict(x_valid).argmax(axis=1)
                )
            )
        step += 1

    return {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_acc": valid_acc
    }


# For a fair comparison, we train the exact same model with the same initialization with the three optimizers below. Feel free to change the hyperparameters if you want.

# In[30]:


model_sgd = kr.models.clone_model(model)
results_sgd = train_model(model_sgd, SGD(model_sgd, learning_rate=1e-2), max_epochs=10, batch_size=256)


# In[31]:


model_sgdm = kr.models.clone_model(model)
results_sgdm = train_model(model_sgdm, SGDMomentum(model_sgdm, learning_rate=1e-1, momentum=0.9), max_epochs=10, batch_size=256)


# In[32]:


model_adam = kr.models.clone_model(model)
results_adam = train_model(model_adam, Adam(model_adam, learning_rate=1e-3), max_epochs=10, batch_size=256)


# In[33]:


x = [256*i for i in range(len(results_sgd['valid_loss']))]
fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=600)
ax[0].plot(results_sgd['train_loss'], color="C0", label='SGD', zorder=2)
ax[0].plot(results_sgdm['train_loss'], color="C1", label='SGD+M', zorder=2)
ax[0].plot(results_adam['train_loss'], color="C2", label='Adam', zorder=2)
ax[0].legend()
ax[0].set_ylabel("Train loss")
ax[0].set_xlabel("Optimizer steps");
ax[0].set_ylim(0, 2.5)
ax[0].grid()
ax[0].legend()

ax[1].scatter(x, results_sgd['valid_loss'], color="C0", s=40, edgecolor='black', label="SGD", zorder=2)
ax[1].scatter(x, results_sgdm['valid_loss'], color="C1", s=40, edgecolor='black', label="SGD+M", zorder=2)
ax[1].scatter(x, results_adam['valid_loss'], color="C2", s=40, edgecolor='black', label="Adam", zorder=2)
ax[1].plot(x, results_sgd['valid_loss'], color="C0", linewidth=2, zorder=2)
ax[1].plot(x, results_sgdm['valid_loss'], color="C1", linewidth=2, zorder=2)
ax[1].plot(x, results_adam['valid_loss'], color="C2", linewidth=2, zorder=2)
ax[1].set_ylim(0, 2.5)
ax[1].set_ylabel("Valid loss")
ax[1].set_xlabel("Optimizer steps");
ax[1].grid()
ax[1].legend();

ax[2].scatter(x, results_sgd['valid_acc'], color="C0", s=40, edgecolor='black', label="SGD", zorder=3)
ax[2].scatter(x, results_sgdm['valid_acc'], color="C1", s=40, edgecolor='black', label="SGD+M", zorder=3)
ax[2].scatter(x, results_adam['valid_acc'], color="C2", s=40, edgecolor='black', label="Adam", zorder=3)
ax[2].plot(x, results_sgd['valid_acc'], color="C0", linewidth=2, zorder=2)
ax[2].plot(x, results_sgdm['valid_acc'], color="C1", linewidth=2, zorder=2)
ax[2].plot(x, results_adam['valid_acc'], color="C2", linewidth=2, zorder=2)
ax[2].legend()
ax[2].grid()
ax[2].set_ylabel("Validation accuracy")
ax[2].set_ylim(0, 1)
ax[2].set_xlabel("Optimizer steps");


# In[34]:


fig, ax = plt.subplots(1, 3, figsize=(14, 6), dpi=600)
ax[0].scatter(x, results_sgd['valid_loss'], color="C0", s=40, edgecolor='black', label="SGD (val)", zorder=3)
ax[0].plot(x, results_sgd['valid_loss'], color="black", zorder=2)
ax[0].plot(results_sgd['train_loss'], color="C0", label='SGD (train)', zorder=1)
ax[0].legend()
ax[0].set_ylim(0, 3)
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Optimizer steps");
ax[0].grid()
ax[0].legend()

ax[1].scatter(x, results_sgdm['valid_loss'], color="C1", s=40, edgecolor='black', label="SGD+M (val)", zorder=3)
ax[1].plot(x, results_sgdm['valid_loss'], color="black", zorder=2)
ax[1].plot(results_sgdm['train_loss'], color="C1", label='SGD+M (train)', zorder=1)
ax[1].legend()
ax[1].set_ylim(0, 3)
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Optimizer steps");
ax[1].grid()
ax[1].legend()

ax[2].scatter(x, results_adam['valid_loss'], color="C2", s=40, edgecolor='black', label="Adam (val)", zorder=3)
ax[2].plot(x, results_adam['valid_loss'], color="black", zorder=2)
ax[2].plot(results_adam['train_loss'], color="C2", label='Adam (train)', zorder=1)
ax[2].legend()
ax[2].set_ylim(0, 3)
ax[2].set_ylabel("Loss")
ax[2].set_xlabel("Optimizer steps");
ax[2].grid()
ax[2].legend();


# Overall accuracy on the whole test set:

# In[35]:


from sklearn.metrics import accuracy_score
x_test, y_test = next(iter(test_ds.batch(10000)))

print("Test accuracies:")
print(f"  SGD     {accuracy_score(y_test, model_sgd.predict(x_test).argmax(axis=1))*100:.2f}%")
print(f"  SGD+M   {accuracy_score(y_test, model_sgdm.predict(x_test).argmax(axis=1))*100:.2f}%")
print(f"  Adam    {accuracy_score(y_test, model_adam.predict(x_test).argmax(axis=1))*100:.2f}%")


# ### Testing optimizers on exotic surfaces

# The result above is that all optimizers perform similarly well with the given model. The differences are too small to find any significant conclusion. However, keep in mind that this can also be attributed to the initialization we chose. When changing the initialization to worse (e.g. constant initialization), Adam usually shows to be more robust because of its adaptive learning rate. To show the specific benefits of the optimizers, we will continue to look at some possible loss surfaces in which momentum and adaptive learning rate are crucial.

# #### Pathological curvatures

# A pathological curvature is a type of surface that is similar to ravines and is particularly tricky for plain SGD optimization. In words, pathological curvatures typically have a steep gradient in one direction with an optimum at the center, while in a second direction we have a slower gradient towards a (global) optimum. Let’s first create an example surface of this and visualize it:

# In[36]:


def plot_surface(f, title, ax, x_range=(-5, 5), y_range=(-15, 5), cmap=cm.viridis):

    # Plot surface on xy plane; choose 3d or 2d plot
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y).numpy()

    # Plot    
    ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=1, color="#000", antialiased=False)
    ax.set_zlabel("loss", rotation=90)
    
    # Formatting plot
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.title(title)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    plt.tight_layout()

    return ax


def plot_contour(f, title, ax, x_range=(-5, 5), y_range=(-15, 5), cmap=cm.viridis):

    # Plot surface on xy plane; choose 3d or 2d plot
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y).numpy()
    
    # Plot
    ax.contourf(X, Y, Z)
    
    # Formatting
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(title)
    ax.set_xlabel(r"$w_1$")
    ax.set_ylabel(r"$w_2$")
    plt.tight_layout()

    return ax


# Consider the function below which has a long, narrow, parabolic shaped flat valley. In terms of optimization, you can image that $w_1$ and $w_2$ are weight parameters, and the curvature represents the loss surface over the space of $w_1$ and $w_2$. Note that in typical networks, we have many, many more parameters than two, and such curvatures can occur in multi-dimensional spaces as well.
# 

# In[37]:


def pathological_curve_loss(w1, w2):
    # Example of a pathological curvature. There are many more possible, 
    # feel free to experiment here!
    x1_loss = kr.activations.tanh(w1)**2 + 0.01 * tf.abs(w1)
    x2_loss = kr.activations.sigmoid(w2)
    return x1_loss + x2_loss

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
plot_surface(pathological_curve_loss, title="Pathological curvature", ax=ax)
plt.show();


# Ideally, our optimization algorithm would find the center of the ravine and focuses on optimizing the parameters towards the direction of $w_2$. However, if we encounter a point along the ridges, the gradient is much greater in $w_1$ than $w_2$, and we might end up jumping from one side to the other. Due to the large gradients, we would have to reduce our learning rate slowing down learning significantly.
# 
# To test our algorithms, we can implement a simple function to train two parameters on such a surface:

# In[38]:


class OptimModel(kr.Model):
    def __init__(self, init):
        super().__init__()
        self.w1 = tf.Variable(init[0], name='w1', trainable=True)
        self.w2 = tf.Variable(init[1], name='w2', trainable=True)


def get_optimizer_path(optim_fn, init, loss_surface, num_steps=100):
    """Plots trajectory of optimizer from initial point (`init`) along a surface
    (`loss_surface`) in minimizing it via gradient updates. Here `optim_func` is
    a function that takes an OptimModel and returns an optimizer. The output of
    this function is a list of 2-tuples which correspond to (x, y) coords of the
    points in its trajectory."""

    model = OptimModel(init)
    optim = optim_fn(model)
    path = [model.get_weights()]
    losses = []

    for _ in range(num_steps):
        with tf.GradientTape() as tape:
            weights = model.trainable_variables
            loss = loss_surface(*weights)

        grads = tape.gradient(loss, weights)
        optim.step(grads)
        path.append(model.get_weights())
        losses.append(loss.numpy())

    return np.array(path), losses


# Compute trajectories of gradient descent for the three optimizers starting from `(-5, 5)`:

# In[39]:


adam = lambda model: Adam(model, learning_rate=1)
sgd  = lambda model: SGD(model, learning_rate=10)
sgdm = lambda model: SGDMomentum(model, learning_rate=10, momentum=0.9)

init = [-5., 5.]
path_adam, losses_adam = get_optimizer_path(adam, init, pathological_curve_loss)
path_sgd,  losses_sgd  = get_optimizer_path(sgd,  init, pathological_curve_loss)
path_sgdm, losses_sgdm = get_optimizer_path(sgdm, init, pathological_curve_loss)

# Get ranges of coordinates of the optimization trajectories + padding
# x = w1 and y = w2, i.e. see ordering on output of get_weights function
all_points = np.concatenate([path_adam, path_sgd, path_sgdm], axis=0)
x_range = (-np.abs(all_points[:, 0]).max(), np.abs(all_points[:, 0]).max())
y_range = (all_points[:, 1].min() - 5, all_points[:, 1].max() + 5)

# Plot surface on box defined by coordinate ranges
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_contour(pathological_curve_loss, "Pathological curvature", ax=ax[0], x_range=x_range, y_range=y_range)

# Plot trajectory of optimizers
ax[0].plot(path_adam[:, 0], path_adam[:, 1], color="red",  marker="o", zorder=1, label="Adam")
ax[0].plot(path_sgd [:, 0], path_sgd [:, 1], color="blue", marker="o", zorder=1, label="SGD")
ax[0].plot(path_sgdm[:, 0], path_sgdm[:, 1], color="gray", marker="o", zorder=1, label="SGD+M")
ax[0].legend()

# Plot loss per optimizer step
ax[1].plot(losses_adam, label='Adam',  color="red")
ax[1].plot(losses_sgd,  label='SGD',   color="blue")
ax[1].plot(losses_sgdm, label='SGD+M', color="gray")
ax[1].legend()
ax[1].set_xlabel("Optimizer step")
ax[1].set_ylabel("Loss")

plt.tight_layout()
plt.show();


# We can clearly see that SGD is not able to find the center of the optimization curve and has a problem converging due to the steep gradients in. In contrast, Adam and SGD with momentum nicely converge as the changing direction of $w_1$ is **canceling itself out**. On such surfaces, it is crucial to use momentum. Indeed, we used a momentum with value $0.9$ which means, the current gradient contributes $0.10$ of its original size.

# #### Steep optima
# 
# A second type of challenging loss surfaces are steep optima. In those, we have a larger part of the surface having very small gradients while around the optimum, we have very large gradients. For instance, take the following loss surfaces:

# In[40]:


def bivar_gaussian(w1, w2, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = (-1 * (w1 - x_mean)**2) / (2 * x_sig**2)
    y_exp = (-1 * (w2 - y_mean)**2) / (2 * y_sig**2)
    return norm * tf.math.exp(x_exp + y_exp)

def comb_func(w1, w2):
    z = -bivar_gaussian(w1, w2, x_mean= 1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= bivar_gaussian(w1, w2, x_mean=-1.0, y_mean= 0.5, x_sig=0.2, y_sig=0.2)
    z -= bivar_gaussian(w1, w2, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
    return z


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
plot_surface(comb_func, x_range=(-2,2), y_range=(-2,2), title="Steep optima", ax=ax)
plt.show();


# Most of the loss surface has very little to no gradients. However, close to the optima, we have very steep gradients. To reach the minimum when starting in a region with lower gradients, we expect an adaptive learning rate to be crucial. To verify this hypothesis, we can run our three optimizers on the surface:

# In[41]:


adam = lambda model: Adam(model, learning_rate=0.2)
sgd  = lambda model: SGD(model, learning_rate=0.5)
sgdm = lambda model: SGDMomentum(model, learning_rate=1, momentum=0.9)

init = [0., 0.]
path_adam, losses_adam = get_optimizer_path(adam, init, comb_func)
path_sgd,  losses_sgd  = get_optimizer_path(sgd,  init, comb_func)
path_sgdm, losses_sgdm = get_optimizer_path(sgdm, init, comb_func)

# Get ranges of coordinates of the optimization trajectories + padding
# x = w1 and y = w2, i.e. see ordering on output of get_weights function
all_points = np.concatenate([path_adam, path_sgd, path_sgdm], axis=0)
x_range = (-1.6, 1.6)
y_range = (-1.6, 1.6)

# Plot surface on box defined by coordinate ranges
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
plot_contour(comb_func, "Steep optima", ax=ax[0, 0], x_range=x_range, y_range=y_range)

# Plot trajectory of optimizers
ax[0, 0].plot(path_adam[:, 0], path_adam[:, 1], color="red",  marker="o", zorder=1, label="Adam")
ax[0, 0].plot(path_sgd [:, 0], path_sgd [:, 1], color="blue", marker="o", zorder=1, label="SGD")
ax[0, 0].plot(path_sgdm[:, 0], path_sgdm[:, 1], color="gray", marker="o", zorder=1, label="SGD+M")
ax[0, 0].legend()

# Plot loss per optimizer step
ax[0, 1].plot(losses_adam, label='Adam',  color="red",  markersize=3.5)
ax[0, 1].plot(losses_sgd,  label='SGD',   color="blue", markersize=3.5)
ax[0, 1].plot(losses_sgdm, label='SGD+M', color="gray", markersize=3.5)
ax[0, 1].legend()
ax[0, 1].set_xlabel("Optimizer step")
ax[0, 1].set_ylabel("Loss")

# Plot x-axis of trajectory of optimizers
ax[1, 0].plot(path_adam[:, 0], color="red",  marker="o", zorder=1, label="Adam",  markersize=3.5)
ax[1, 0].plot(path_sgd [:, 0], color="blue", marker="o", zorder=1, label="SGD",   markersize=3.5)
ax[1, 0].plot(path_sgdm[:, 0], color="gray", marker="o", zorder=1, label="SGD+M", markersize=3.5)
ax[1, 0].legend()
ax[1, 0].grid()
ax[1, 0].set_xlabel("steps")
ax[1, 0].set_ylabel(r"$w_1$")
ax[1, 0].set_ylim(-5, 1)

# Plot y-axis of trajectory of optimizers
ax[1, 1].plot(path_adam[:, 1], color="red",  marker="o", zorder=1, label="Adam",  markersize=3.5)
ax[1, 1].plot(path_sgd [:, 1], color="blue", marker="o", zorder=1, label="SGD",   markersize=3.5)
ax[1, 1].plot(path_sgdm[:, 1], color="gray", marker="o", zorder=1, label="SGD+M", markersize=3.5)
ax[1, 1].legend()
ax[1, 1].grid()
ax[1, 1].set_xlabel("steps")
ax[1, 1].set_ylabel(r"$w_2$")
ax[1, 1].set_ylim(-7, 1)

plt.tight_layout()
plt.show();


# SGD first takes very small steps until it touches the border of the optimum. First reaching a point around $(-0.5,-0.9)$, the gradient direction has changed and pushes the parameters to $(0.5, 0.9)$ from which SGD cannot recover anymore (only with many, many steps). A similar problem has SGD with momentum, only that it continues the direction of the touch of the optimum. The gradients from this time step are so much larger than any other point that the momentum ${\boldsymbol m}_t$ is overpowered by it despite having the factor $(1-\beta_1) = 0.1.$ Finally, Adam is able to converge in the optimum showing the importance of adaptive learning rates.

# ### What optimizer to take
# 
# After seeing the results on optimization, what is our conclusion? Should we always use Adam and never look at SGD anymore? The short answer: no. There are many papers saying that in certain situations, SGD (with momentum) generalizes better where Adam often tends to overfit [[1](https://proceedings.neurips.cc/paper/2017/file/81b3833e2504647f9d794f7d7b9bf341-Paper.pdf), [2](https://arxiv.org/abs/1609.04747)]. This is related to the idea of finding wider optima. For instance, see the illustration of different optima below (from [Keskar et al., 2017](https://arxiv.org/pdf/1609.04836.pdf)):
# 
# ```{figure} ../../img/flat_vs_sharp_minima.svg
# ```
# 
# The black line represents the training loss surface, while the dotted red line is the test loss. Finding sharp, narrow minima can be helpful for finding the minimal training loss. However, this doesn't mean that it also minimizes the test loss as especially flat minima have shown to generalize better. You can imagine that the test dataset has a slightly shifted loss surface due to the different examples than in the training set. A small change can have a significant influence for sharp minima, while flat minima are generally more robust to this change. 
# 
# In the notebook [*Inception, ResNet, and DenseNet*](https://particle1331.github.io/inefficient-networks/notebooks/tensorflow/06-tensorflow-inception.html), we will see that some network types can still be better optimized with SGD and learning rate scheduling than Adam. Nevertheless, Adam is the most commonly used optimizer in Deep Learning as it usually performs better than other optimizers, especially for deep networks.

# ## Conclusion
# 
# In this notebook, we have looked at initialization and optimization techniques for neural networks. We have seen that a good initialization has to balance the preservation of the gradient variance as well as the activation variance. This can be achieved with the Xavier initialization for tanh-based networks, and the Kaiming initialization for ReLU-based networks. In optimization, concepts like momentum and adaptive learning rate can help with challenging loss surfaces but don’t guarantee an increase in performance for neural networks.
# 
