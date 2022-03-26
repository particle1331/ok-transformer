#!/usr/bin/env python
# coding: utf-8

# # Mechanics of TensorFlow

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

# In this notebook, we take a deeper dive into lower-level features of TensorFlow. For example, accessing and modifying layer weights and weight gradients, performing automatic differentiation, creating custom layers, and so on. Knowing these tricks would allow us to write more advanced TensorFlow models and write custom functionality for our models and training pipeline. 

# In[1]:


import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# plotting
import matplotlib.pyplot as plt 
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')


# ```{margin}
# ⚠️ **Attribution:** This notebook builds on the [Chapter 14 notebook](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/ch14) of {cite}`RaschkaMirjalili2019` which is [released under MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt). For instance, we add timing static graph executions to see the conditions where we can expect a speedup in our code. 
# ```

# ## Static graph execution

# Computations with eager execution are not as efficient 
# as the static graph execution in TensorFlow v1.x, as these can come with pure Python operations. 
# TensorFlow v2 provides a tool called **AutoGraph** that can automatically transform Python code into 
# TensorFlow's graph code for faster execution. Fortunately for us, TensorFlow provides 
# a simple mechanism for compiling a normal Python function to do exactly this: `graph_function = tf.function(eager_function)` or using the `@tf.function` decorator.

# ### Specifying input signature and static graph tracing 

# Note that while TensorFlow graphs, strictly speaking, require static types and shapes, 
# `tf.function` readily supports such a dynamic typing capability (through separate static graphs will be created under the hood). For example, let's call this function 
# with the following inputs:

# In[2]:


def f(x, y, z):
    return x + y + z

f_graph = tf.function(f)
tf.print('Scalar Inputs:', f_graph(1, 2, 3))
tf.print('Rank 1 Inputs:', f_graph([1], [2], [3]))
tf.print('Rank 2 Inputs:', f_graph([[1]], [[2]], [[3]]))


# Here, TensorFlow uses a **tracing mechanism** to construct a graph based on the input arguments. For this tracing mechanism, TensorFlow generates a tuple of keys based on the input signatures
# given for calling the function. The generated keys are as follows:
# * For `tf.Tensor` arguments, the key is based on their shapes and `dtypes`.
# * For Python types, such as lists, their `id()` is used to generate cache keys.
# * For Python primitive values, the cache keys are based on the input values.

# Upon calling such a decorated function, TensorFlow will check whether a graph with
# the corresponding key has already been generated. If such a graph does not exist,
# TensorFlow will generate a new graph and store the new key. 
# If we want to limit the way a function can be called, we can specify its input signature
# via a tuple of `tf.TensorSpec` objects when defining the function. For example, let's
# take the previous function and modify it so that only rank 1 tensors of
# type `tf.int32` are allowed:

# In[3]:


def f(x, y, z):
    return x + y + z

f_graph = tf.function(f, input_signature=(
    tf.TensorSpec(shape=[None], dtype=tf.int32),
    tf.TensorSpec(shape=[None], dtype=tf.int32),
    tf.TensorSpec(shape=[None], dtype=tf.int32)
    )
)

tf.print('Rank 1 Inputs:', f_graph([1], [2], [3]))


# We get an error if we pass a tensor with different shape:

# In[4]:


try:
    tf.print('Rank 1 Inputs:', f_graph([[1]], [[2]], [[3]]))
except Exception as e:
    print(e)


# ### Timing static execution runs

# In this section, we define a function that takes in an eager function `f` and plots the timings for evaluating `f` eagerly on `x` versus evaluating on its static graph version `tf.function(f)`. As discussed above, the static graph is built after "warming up" once on `x` with its particular shape and type through TensorFlow's tracing mechanism.

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit

def compare_timings(f, x, n):
    # Define functions
    eager_function = f
    graph_function = tf.function(f)

    # Timing
    graph_time = timeit(lambda: graph_function(x), number=n)
    eager_time = timeit(lambda: eager_function(x), number=n)
    
    return {
        'graph': graph_time,
        'eager': eager_time
    }


# ```{margin}
# For further info on TF 2.x's tracing mechanism, refer to [this guide](https://www.tensorflow.org/guide/function#tracing).
# ```

# Note that if we fail to persist the static graph, we get the following warning, as well as practically an endless loop. The error message also highlights cases where we make inefficient use of tracing. (Recall the rules above for generating keys for static graphs based on the input.)
# 
# > WARNING:tensorflow:6 out of the last 6 calls to <keras.engine.sequential.Sequential object at 0x28575dfa0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop.

# Comparing static graph execution with eager execution on a dense network:

# In[6]:


from tensorflow.keras.layers import Flatten, Dense

# Model building
model = tf.keras.Sequential()
model.add(Flatten())
model.add(Dense(256, "relu"))
model.add(Dense(256, "relu"))
model.add(Dense(256, "relu"))
model.add(Dense(10, "softmax"))

# Define input + functions
u = tf.random.uniform([100, 28, 28])
mlp_times = compare_timings(model, u, n=10000);


# Comparing timings with convolution operations:

# In[7]:


from tensorflow.keras.layers import Conv2D, AveragePooling2D

# Build model
conv_model = tf.keras.Sequential()
conv_model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
conv_model.add(AveragePooling2D())
conv_model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
conv_model.add(AveragePooling2D())

# Plot timings
u = tf.random.uniform([100, 28, 28, 1])
conv_times = compare_timings(conv_model, u, n=10000);


# Comparing timings on many extremely small operations:

# In[8]:


def small_dense_layer(x):
    w = tf.random.uniform(shape=(3, 1), dtype=tf.float16)
    b = tf.random.uniform(shape=(1,), dtype=tf.float16)
    return tf.add(tf.matmul(x, w), b)

# Plot timings
x = tf.random.uniform(shape=(1, 3), dtype=tf.float16)
small_times = compare_timings(small_dense_layer, x, n=10000);


# In[9]:


models = ['mlp','conv','small']
eager = [eval(m + '_times')['eager'] for m in models]
graph = [eval(m + '_times')['graph'] for m in models]
x = np.arange(len(models))


plt.bar(x + 0.1, eager, width=0.2, label='eager')
plt.bar(x - 0.1, graph, width=0.2, label='graph')
plt.xticks(x, models)
plt.ylabel("Time (s)")
plt.title("Time for 10,000 executions.")
plt.legend();


# The above results show that graph execution can be faster can be faster than eager code, especially for graphs with expensive operations. But for graphs with few expensive operations (like convolutions), you may not see much speedup or even worse with many cheap operations. Perhaps because there is overhead in the tracing mechanism for static graphs and alternating between TensorFlow and Python abstractions.

# ## TensorFlow `Variable`

# A `Variable` is a special `Tensor` object
# that allows us to store and update the parameters of our models during training.
# This can be created by just calling the `tf.Variable` class on user-specified
# initial values. 

# In[10]:


a = tf.Variable(initial_value=3.0, name='var_a') # float32 by default.
b = tf.Variable(initial_value=[1, 2, 3], name='var_b')
c = tf.Variable(initial_value=['c'], dtype=tf.string)

print(a)
print(b)
print(c)


# Note that initial value is required. TF variables have an attribute called `trainable`, which by default is set to `True`. Higher-level APIs such as Keras will use this attribute to manage the trainable variables and non-trainable ones. You can define a non-trainable `Variable` as follows:

# In[11]:


w = tf.Variable(3.0, trainable=False)
print(w.trainable)


# ### Modifying the value of a variable

# The values of a `Variable` can be efficiently modified by running some operations
# such as `.assign()`, `.assign_add()` and related methods. When the `read_value` argument is set to `True` (default), these operations will automatically return the new values after updating the current values of the `Variable`.

# In[12]:


w.assign(0.0, read_value=True)


# Setting the `read_value` to `False` will suppress the automatic return of the updated value but the `Variable` will still be updated in place.

# In[13]:


tf.print(w.assign_add(-1.0, read_value=False))
tf.print(w)


# ### Initializing a TensorFlow module

# In practice, we usually define and initialize a `Variable` inside a `tf.Module` class. In the example below, we define two variables one trainable one and another non-trainable. These variables can be accessed using the `.variables` and `.trainable_variables` attribute of TF modules.

# In[14]:


class MyModule(tf.Module):
    def __init__(self):
        init = tf.keras.initializers.GlorotNormal()
        self.w1 = tf.Variable(init(shape=(2, 3)), trainable=True)
        self.w2 = tf.Variable(init(shape=(1, 2)), trainable=False)


m = MyModule()
print("All module variables:", [v.shape for v in m.variables])
print("Trainable variables:", [v.shape for v in m.trainable_variables])


# ### Variables and TF functions
# 
# Note that if we define a TF variable inside a pure Python function, then this variable will be initialized every time the function is called. Since the static graph will try to reuse the variable based on tracing and graph creation, TF prevents variable initialization inside a decorated TF function. 
# 
# One way to avoid this problem is to define the `Variable` outside of the decorated
# function and use it inside the function &mdash; this is not recommended with a global scope. Instead, you should define a class to manage this dependency in a separate namespace.

# In[15]:


@tf.function
def f(x):
    w = tf.Variable([3.0])
    return x * w

# Testing
try:
    f(1.0)
except Exception as e:
    print(e)


# Instead do:

# In[16]:


# Declare variable outside function <- make sure to not pollute the global namespace
w = tf.Variable([3.0])

@tf.function
def f(x):
    return x * w

# Testing
try:
    tf.print(f(1.0))
except Exception as e:
    print(e)


# ## Automatic Differentiation

# TensorFlow supports automatic differentiation which implements symbolic differentiation for each operation defined in the language. For nested functions, TF provides a context called `GradientTape` for calculating gradients of these computed tensors with respect to its dependent nodes in the computation graph. This allows TensorFlow needs to remember what operations happen in what order during the forward pass. This list of operations is traversed backwards during [backward pass](https://particle1331.github.io/inefficient-networks/notebooks/fundamentals/backpropagation.html) to compute the weight gradients.

# ### Gradients of the loss with respect to weights

# In order to compute these gradients, we have to "record" the computations via `tf.GradientTape`. Note that the shape of `tape.gradient(loss, w)` is the same as that of `w`.  

# In[17]:


# scope outside tf.GradientTape
w = tf.Variable(1.0)
b = tf.Variable(0.5)

# data
x = tf.convert_to_tensor([1.4])
y = tf.convert_to_tensor([2.1])

with tf.GradientTape() as tape:
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

grad = tape.gradient(loss, w)
tf.print("∂(loss)/∂w =", grad)
tf.print(2 * (y - w*x - b) * (-x)) # symbolic


# #### Higher-order gradients

# It turns out that TF supports stacking gradient tapes which allow us to compute **second derivatives**:

# In[18]:


with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        z = tf.add(tf.multiply(w, x), b)
        loss = tf.reduce_sum(tf.square(y - z))
    grad_w = inner_tape.gradient(loss, w)
grad_wb = outer_tape.gradient(grad_w, b)

tf.print("∂²(loss)/∂w∂b =", grad_wb)
tf.print(2 * (-1) * (-x)) # symbolic


# ### Gradients with respect to nontrainable parameters

# `GradientTape` automatically supports the gradients for trainable variables.
# For non-trainable variables[^refadversarial] and other `Tensor` objects, we need to add
# `tape.watch()` to monitor those as well.
# 
# [^refadversarial]: Computing gradients of the loss with respect to the input
# example is used for generating adversarial examples.

# In[19]:


with tf.GradientTape() as tape:
    tape.watch(x) # <-
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

grad = tape.gradient(loss, x)
tf.print("∂(loss)/∂x =", grad)
tf.print(2 * (y - w*x - b) * (-w)) # check symbolic


# ### Persisting the gradient tape

# Note that the tape will keep the resources only for a single gradient computation by default. So
# after calling `tape.gradient()` once, the resources will be released and the tape will
# be cleared. If we want to compute more than one gradient, we need to persist it (less memory efficient).

# In[20]:


with tf.GradientTape() as tape:
    tape.watch(x)
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

try:
    tf.print("∂(loss)/∂w =", tape.gradient(loss, w))
    tf.print("∂(loss)/∂x =", tape.gradient(loss, x))
except Exception as e:
    print(e)


# In[21]:


with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    z = tf.add(tf.multiply(w, x), b)
    loss = tf.reduce_sum(tf.square(y - z))

tf.print("∂(loss)/∂w =", tape.gradient(loss, w))
tf.print("∂(loss)/∂x =", tape.gradient(loss, x)) # grad_x has same shape as x


# ### Applying optimizer step to update model parameters

# During SGD, we are computing gradients of a loss term with respect to model weights, which we use to update the weights according to some rule defined by an optimization algorithm. For Keras optimizers, we can do this by  using `.apply_gradients`:

# In[22]:


grad_w = tape.gradient(loss, w)
grad_b = tape.gradient(loss, b)
lr = 0.1
tf.print('w =', w)
tf.print('b =', b)
tf.print('λ =', lr)
tf.print('grad_[w, b] =', [grad_w, grad_b])

# Define keras optimizer; apply optimizer step
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
optimizer.apply_gradients(zip([grad_w, grad_b], [w, b]))

# Print updates
tf.print()
tf.print('w - λ·∂(loss)/∂w ≟', w)
tf.print('b - λ·∂(loss)/∂b ≟', b)


# Checks out nicely.

# ## Keras API

# Keras provides a user-friendly and
# modular programming interface that allows easy prototyping and the building of
# complex models in just a few lines of code which
# in TensorFlow 2, has become the primary and recommended approach
# for implementing models via the `tf.keras` library.  This has the advantage that it supports TensorFlow specific functionalities, such as [dataset pipelines using `tf.data`](https://particle1331.github.io/inefficient-networks/notebooks/tensorflow/01-tensorflow-nn.html#).

# ### Stacking layers with `tf.keras.Sequential`

# For stacking layers that perform sequential transforms on input data, we typically use `tf.keras.Sequential()`. Such a model has an `add()` method to add individual Keras layers. Alternatively, if we have a list of layers `layers`, then we can define a model using `tf.keras.Sequential(layers)`.

# In[23]:


from tensorflow.keras.layers import Dense

# Create model
model = tf.keras.Sequential()
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='relu'))

# Build model
model.build(input_shape=(None, 4))
model.summary()


# Once variables (or model parameters) are created, we can access both
# trainable and non-trainable variables as follows:

# In[24]:


for v in model.variables:
    print(f'{v.name:20s} {str(v.trainable):7} {v.shape}')


# ### Regularization and initialization

# Layers can be configured using optional arguments to allow applying different activation
# functions, choosing variable initializers, or choosing the type regularization to use.
# Regularizers allow you to apply penalties on layer parameters or layer activity during optimization. These penalties are summed into the loss function that the network optimizes. For Keras, regularization penalties are applied on a per-layer basis. The `Dense` layer (as well as other layers such as `Conv1D`, `Conv2D`, and `Conv3D`) exposes three keyword arguments:
# 
# * `kernel_regularizer`: Regularizer to apply a penalty on the layer's kernel
# * `bias_regularizer`: Regularizer to apply a penalty on the layer's bias
# * `activity_regularizer`: Regularizer to apply a penalty on the layer's output
# 
# <br>

# In[25]:


import tensorflow.keras.initializers as init
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.optimizers as optim


# Create model
model = tf.keras.Sequential()
model.add(
    Dense(
        units=16, 
        activation='relu',
        kernel_initializer=init.glorot_uniform(),
        bias_initializer=init.Constant(2.0),
        kernel_regularizer=regularizers.l1
    )
)

# Build model
model.build(input_shape=(None, 4))
model.summary()


# ### Compiling Keras models

# Compiling prepares the model for training via `model.fit()`. In this example, we will compile the model using the SGD optimizer, cross-entropy
# loss for binary classification, and a specific list of metrics, including accuracy,
# precision, and recall:

# In[26]:


model.compile(
    optimizer=optim.SGD(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.Accuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)


# #### API for losses
# 
# APIs for cross-entropy loss are always tricky. For example, libraries always provide computation of the cross-entropy loss by providing the logits, instead of the class-membership probabilities. This is usually preferred due to numerical stability reasons and can be implemented by setting `from_logits=True`. You can see how this is possible by working out the math (some operations cancel out). The default behavior in Keras is `from_logits=False`, so that the outputs of the model are expected to be probabilities in $[0, 1].$

# ```{figure} https://raw.githubusercontent.com/rasbt/python-machine-learning-book-3rd-edition/master/ch15/images/15_11.png
# ---
# name: loss-logits-keras
# width: 45em
# ---
# 
# Keras API for loss functions. {cite}`RaschkaMirjalili2019` (Chapter 15)
# ```

# ### Solving the XOR problem

# **Dataset.** The XOR is the smallest dataset that is not linearly separable (also the most historically interesting relative to its size {cite}`Minsky1969`). Our version of the XOR dataset is generated by adding Gaussian noise to points `(-1, -1)`, `(-1, 1)`, `(1, -1)` and `(1, 1)`. Points generated from `(1, 1)` and `(-1, -1)` will be labeled `1` otherwise `0`. A dataset of size 200 points will be generated with half used for validation.

# In[27]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

# Create dataset
X = []
Y = []
for p in [(1, 1), (-1, -1), (-1, 1), (1, -1)]:
    x = np.array(p) + np.random.normal(0, 0.3, size=(50, 2)) 
    y = int(p[0] * p[1] > 0) * np.ones(50)
    X.append(x)
    Y.append(y)

X = np.concatenate(X)
Y = np.concatenate(Y)

# Train-test split
indices = list(range(200))
np.random.shuffle(indices)
valid = indices[:100]
train = indices[100:]
X_valid, y_valid = X[valid, :], Y[valid]
X_train, y_train = X[train, :], Y[train]

# Visualize dataset
fig = plt.figure(figsize=(6, 6))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], s=40, edgecolor='black', label=0)
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], s=40, edgecolor='black', label=1)
plt.legend();


# <br>
# 
# **Network architecture.** As a general rule of thumb, the more layers we have,
# and the more neurons we have in each layer, the larger the capacity of the model
# will be. While having more parameters means the network can fit more complex functions, larger models are usually harder to train (and are prone to overfitting). Model capacity can be increased by increasing:
# 
# * **Width.** 
# The universal approximation theorem states that a feedforward NN with a single hidden 
# layer and a sufficiently large number of hidden units can approximate any continuous function
# to arbitrary accuracy. But this doesn't apply to the current task which is a classification problem. Indeed, we need a network of at least depth 2 to properly classify the dataset.
# <br><br>
# 
# * **Depth.** The advantage of making a network deeper rather than wider is 
# that fewer parameters are required to achieve a comparable model capacity. 
# However, a downside of deep (versus wide) models is that deep models are prone
# to vanishing and exploding gradients, which make them harder to train.
# <br><br>

# As mentioned, from the geometry of the dataset, we have to use a network that has at least depth 2, so its not linear. However, since the dataset is small, we want the network to be not too wide (and not too deep), so the model does not overfit on the dataset.

# In[28]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=4, activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Keras provides `.summary()` which prints a summary of the network architecture.
# Since the number of parameters for the input layer depend on input size , we 
# need to specify the dimension of the input. This is done by calling `model.build`
# on the expected input shape:

# In[29]:


model.build(input_shape=(None, 2))
model.summary()


# `None` is used as a placeholder for the first dimension of the input to make room for arbitrary batch sizes. Alternatively, we could have set `input_shape` in the input layer so we can skip model build. 

# <br>
# 
# **Model training.** Writing the `train()` function is boilerplate code.
# Since the training loop can be complex, developing this each time can
# potentially be a source of bugs, not to mention time-consuming. 
# TensorFlow Keras API provides a convenient `.fit()` method that can be called 
# on an instantiated model.

# In[30]:


model.compile(
    optimizer=optim.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

hist = model.fit(
    X_train, y_train, # fit accepts numpy arrays
    validation_data=(X_valid, y_valid),
    epochs=200,
    batch_size=2, verbose=0
)


# ```{margin}
# Refer to the [Keras developer guides](https://keras.io/guides/), if more precise control of the details of the training process is needed.
# ```
# 
# The `fit` method handles the low-level details (regularization, callbacks, metrics, etc.) of training consistently across different implementations. Moreover, this is designed to be performant by exploiting static graph computation. Hence, it is recommended to use `fit` for most use-cases (as well as other built-ins such as `evaluate` and `predict` for inference).

# <br>
# 
# **Results.** The `fit()` method returns a dictionary containing data on how the model trained. We will use this to generate visualizations of the training process. To further evaluate the model, we also look at its decision boundaries. 

# In[31]:


hist.history.keys()


# In[32]:


from mlxtend.plotting import plot_decision_regions

def plot_training_history(hist):
    _, ax = plt.subplots(1, 3, figsize=(12, 3))

    ax[0].plot(range(200), hist.history['loss'], label='train loss')
    ax[0].plot(range(200), hist.history['val_loss'], label='valid loss')
    ax[0].legend()

    ax[1].plot(range(200), hist.history['binary_accuracy'], label='train acc.')
    ax[1].plot(range(200), hist.history['val_binary_accuracy'], label='valid acc.')
    ax[1].legend();

    ax[2] = plot_decision_regions(X=X_valid, y=y_valid.astype(np.int_), clf=model, legend=2)
    return ax

model.evaluate(X_valid, y_valid)
plot_training_history(hist);


# To see model confidence, we can look at the prediction probability at each point in $[-1, 1]^2.$

# In[33]:


from matplotlib.colors import to_rgba

# Plot valid set points
def plot_decision_gradient(model):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(X_valid[y_valid==0, 0], X_valid[y_valid==0, 1], s=40, edgecolor='black', label=0)
    plt.scatter(X_valid[y_valid==1, 0], X_valid[y_valid==1, 1], s=40, edgecolor='black', label=1)

    # Plot decision gradient
    c0 = np.array(to_rgba("C0"))
    c1 = np.array(to_rgba("C1"))
    x1 = np.arange(-2, 2, step=0.01)
    x2 = np.arange(-2, 2, step=0.01)

    xx1, xx2 = np.meshgrid(x1, x2)
    model_inputs = np.stack([xx1, xx2], axis=-1)
    preds = model(model_inputs.reshape(-1, 2)).numpy().reshape(400, 400, 1)
    output_image = (1 - preds) * c0 + preds * c1 # blending
    plt.imshow(output_image, origin='lower', extent=(-2, 2, -2, 2));

# Plotting
plot_decision_gradient(model);


# ### Keras' Functional API

# Recall that using `Sequential` only allows for a sequence of transformations. This is too restrictive for other architectures. Keras' so-called functional API comes in handy for more complex transformations such as residual connections. Observe that the model build adds a new "layer" called `tf.__operators__.add`.

# In[34]:


# Specify input and output
x = tf.keras.Input(shape=(2,))
f = Dense(units=2, input_shape=(2,), activation='relu')(x)
out = Dense(units=1, activation='sigmoid')(x + f)

# Build model
model = tf.keras.Model(inputs=x, outputs=out)
model.summary() # compile, fit, etc. also works 


# ### Keras' `Model` class

# An alternative way to build complex models is by subclassing `tf.keras.Model`. A model derived from `tf.keras.Model` inherits methods such as `build()`, `compile()`, and `fit()`.

# In[35]:


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(units=4, activation='tanh')
        self.dense2 = Dense(units=1, activation='sigmoid')

    def call(self, x):
        h1 = self.dense1(x)
        out = self.dense2(h1)
        return out


# Build model and model summary
model = MyModel()
model.build(input_shape=(None, 2))
model.summary()


# ### Creating custom Keras layers

# Notice that we've been using Keras layers in defining our models. In cases where we want to define a new layer that is not already supported by Keras, or customizing an existing layer,
# we can do this by extending the `Layer` base class. In the custom class, we have to define `__init__()` and `call()`. The `build()` method handles delayed variable initialization. Finally, we define a `get_config()` which can be useful for model serialization (saving and loading). 

# <br>
# 
# **Implementation.** To illustrate the concept of implementing custom layers, let's consider a simple
# example. We define a new linear layer that computes $(\mathbf x + \boldsymbol{\varepsilon}) \cdot \mathbf w + \boldsymbol {b}$
# where $\boldsymbol\varepsilon$ refers to a random variable as noise, then passes the result to a ReLU nonlinearity. We assume that $\mathbf x$ is a rank 2 tensor with shape `(B, d)` where `B` is the batch size and `d` is the size of the (flattened) inputs. 

# In[36]:


class NoisyLinear(tf.keras.layers.Layer):
    def __init__(self, output_dim, noise_stddev=0.1, **kwargs):
        self.output_dim = output_dim
        self.noise_stddev = noise_stddev
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name='weights', 
            shape=(input_shape[1], self.output_dim),
            initializer='random_normal',
            trainable=True
        )

        self.b = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, x, training=False):
        if training:
            noise = tf.random.normal(
                shape=x.shape, 
                mean=0.0, 
                stddev=self.noise_stddev
            )
        else:
            noise = tf.zeros_like(x)
        
        z = tf.matmul(x + noise, self.w) + self.b
        return tf.keras.activations.relu(z)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "noise_stddev": self.noise_stddev
        })
        return config


# ```{margin}
# This is analogous to `model.train()` and `model.eval()` in PyTorch. Though, I think having an explicit variable to handle this is nice.
# ```
# 
# Notice that in the `call()` method, we have used an
# additional argument, `training=False`. This distinguishes whether a model or layer
# is used at training or at inference time. This is automatically set in Keras to `True` when using `.fit` and to `False` when using `.predict`. The `training` flag is implemented there are operations 
# that behave differently in
# training and prediction modes such as dropout and batch normalization. In the case of `NoisyLayer`, noise is only added during training; no noise is added at inference. 

# In[37]:


noisy_layer = NoisyLinear(output_dim=1)
noisy_layer.build(input_shape=(None, 4))

tf.print(noisy_layer.w)
tf.print(noisy_layer.b)


# Let's look at the outputs:

# In[38]:


noise = []
noisy_layer = NoisyLinear(output_dim=1)

x = tf.zeros(shape=(1, 1))
for i in range(10000):
    noise.append((noisy_layer(x, training=True) - noisy_layer.b) / noisy_layer.w)

# Plot distribution
import seaborn as sns
sns.distplot(noise);


# Looks OK, the ReLU clips the output to the positive x-axis. Removing zero, we expect the distribution to look like a half-Gaussian.

# In[39]:


sns.distplot([t for t in noise if t.numpy()[0, 0] > 0]);


# Testing the `.config` method.

# In[40]:


# Re-building from config:
config = noisy_layer.get_config()
new_layer = NoisyLinear.from_config(config)
print(config)

# Output can be different since random state is not saved
x = tf.zeros(shape=(1, 1))
tf.print(noisy_layer(x, training=True))
tf.print(new_layer(x, training=True))


# Testing call outside training:

# In[41]:


s = 0
for i in range(10):
    s += noisy_layer(tf.zeros(shape=(1, 1)))

print(s)


# <br>
# 
# **Remodelling.** In this section, we will add `NoisyLinear` to our previous model for XOR. Note that noise should be scaled depending on the magnitude of the input. In our case, the input features vary between $-1$ to $1,$ so we set $\sigma = 0.3$ in the noisy linear for it to have considerable effect.

# In[42]:


model = tf.keras.Sequential()
model.add(NoisyLinear(output_dim=4, noise_stddev=0.3))
model.add(tf.keras.layers.Dense(units=4, input_shape=(2,), activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.build(input_shape=(None, 2))
model.summary()


# In[43]:


model.compile(
    optimizer=optim.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

hist = model.fit(
    X_train, y_train, # fit accepts numpy arrays
    validation_data=(X_valid, y_valid),
    epochs=200,
    batch_size=2, verbose=0
)


# In[44]:


plot_training_history(hist);


# In[45]:


model.evaluate(X_valid, y_valid)


# Notice that the training curve is noisier than before since we added a large amount of noise. On the other hand, the validation curves are not noisy at all. This shows Keras automatically sets `training` to `False` during evaluation. Also note that while the validation performance is perfect, it seems to generalize worse since the decision boundaries are too sharp.

# In[46]:


plot_decision_gradient(model); # surprisingly sharp


# ### Saving and loading models

# We can save and load a model for checkpointing as follows:

# In[47]:


model.save('model.h5', 
    overwrite=True, 
    include_optimizer=True, # also save state of optimizer 
    save_format='h5'
)


# **Testing load.** Note that custom layers need to be taken particular care of.

# In[48]:


model_load = tf.keras.models.load_model(
    'model.h5', 
    custom_objects={'NoisyLinear': NoisyLinear}
)

model_load.summary()


# We can also save the network architecture as a JSON file.

# In[49]:


import json

json_object = json.loads(model_load.to_json())
print(json.dumps(json_object, indent=2))


# In[ ]:




