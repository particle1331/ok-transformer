#!/usr/bin/env python
# coding: utf-8

# # Introduction to PyTorch
# 

# ```{admonition} Attribution
# This notebook is based on [Tutorial 2](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial2/Introduction_to_PyTorch.html) of the [Deep Learning Course](https://uvadlc.github.io/lectures-nov2020.html#) at the University of Amsterdam. The full list of tutorials can be found [here](https://uvadlc-notebooks.rtfd.io).
# ```

# The following notebook is meant to give a short introduction to PyTorch basics, and get you setup for writing your own neural networks. **PyTorch** is an open source machine learning framework that allows you to write your own neural networks and optimize them efficiently. However, PyTorch is not the only framework of its kind. Alternatives to PyTorch include TensorFlow, JAX and MXNet. We choose PyTorch because it is well established, has a huge developer community (originally developed by Facebook), is very flexible and especially used in research. Many current papers publish their code in PyTorch, and thus it is good to be familiar with PyTorch as well. 

# In[1]:


import os
import math
import numpy as np 
import time

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from tqdm import tqdm


# ## Basics of PyTorch
# 
# We will start with reviewing the very basic concepts of PyTorch. As a first step, let us check the version of the installed PyTorch library `torch`: 

# In[2]:


import torch
print("Using torch", torch.__version__)


# ```{margin}
# **Setting the seed**
# ```
# 
# As in every machine learning framework, PyTorch provides functions that are stochastic like generating random numbers. However, a very good practice is to setup your code to be reproducible with the exact same random numbers. This is why we set a seed below. 

# In[3]:


torch.manual_seed(42) # Setting the seed


# ### Tensors
# 
# ```{margin}
# **Tensors = numpy arrays + GPU support**
# ```
# 
# Tensors are the PyTorch equivalent to NumPy arrays, with the addition to also have support for GPU acceleration (more on that later).
# Hence, as a prerequisite, we recommend to be familiar with the `numpy` package as most machine learning frameworks are based on very similar concepts. In this case, the name "tensor" is a generalization of concepts you already know. For instance, a vector is a 1-D tensor, and a matrix a 2-D tensor. When working with neural networks, we will use tensors of various shapes and number of dimensions.
# 
# Most common functions you know from NumPy can be used on tensors as well. Actually, since NumPy arrays are so similar to tensors, we can convert most tensors to NumPy arrays (and back) but we don't need it too often.
# 
# #### Initialization
# 
# Let's first start by looking at different ways of creating a tensor. There are many possible options, the most simple one is to call `torch.Tensor` with the desired shape as input argument:

# In[4]:


x = torch.Tensor(2, 3, 4)
print(x)


# The constructor `torch.Tensor` is equivalent to `torch.empty` which allocates memory for the desired tensor, but reuses any values that have already been in the memory. To create a tensor from data we can use `torch.tensor(data, dtype=None)` which returns a tensor version of the data with the appropriate type. 

# In[5]:


# Create a tensor from data
x = torch.tensor([[1, 2], [3, 4]])
print(x)


# For tensors with specific values, e.g. ones and zeros, we can use `torch.zeros` and `torch.ones` instead of passing explicit data. Similarly, we can use `torch.rand` to sample from the uniform distribution on $[0, 1]$, and `torch.randn` to sample from the standard normal. These functions take in `*args` such that the resulting tensor has shape `args`. Finally, `torch.arange(m, n, step=h)` initializes a tensor of equally spaced numbers in `[m, n)` with step-size `h`.

# In[6]:


# Create a tensor with random values between 0 and 1 with the shape [2, 3, 4]
x = torch.rand(2, 3, 4)
print(x)


# You can obtain the shape of a tensor in the same way as in NumPy (`x.shape`), or using the `.size` method:

# In[7]:


shape = x.shape
print("Shape:", x.shape)

size = x.size()
print("Size:", size)

dim1, dim2, dim3 = x.size()
print("Size:", dim1, dim2, dim3)


# #### Converting: tensor $\leftrightarrow$ numpy
# 
# Tensors can be converted to numpy arrays, and numpy arrays back to tensors. To transform a numpy array into a tensor, we can use the function `torch.from_numpy`:

# In[8]:


np_arr = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(np_arr)

print("NumPy array:") 
print(np_arr)
print("\nPyTorch tensor:") 
print(tensor)


# To transform a PyTorch tensor back to a numpy array, we can use the function `.numpy()` on tensors:

# In[9]:


tensor = torch.arange(4)
np_arr = tensor.numpy()

print("PyTorch tensor:", tensor)
print("NumPy array:", np_arr)


# The conversion of tensors to numpy require the tensor to be on the CPU, and not the GPU (more on GPU support in a later section). In case you have a tensor on GPU, you need to call `.cpu()` on the tensor beforehand. Hence, you get a line like `np_arr = tensor.cpu().numpy()`.

# #### Operations
# 
# Most operations that exist in numpy, also exist in PyTorch. A full list of operations can be found in the [PyTorch documentation](https://pytorch.org/docs/stable/tensors.html#), but we will review the most important ones here.
# 
# The simplest operation is to add two tensors:

# In[10]:


x1 = torch.rand(2, 3)
x2 = torch.rand(2, 3)
y = x1 + x2

print("x1 =")
print(x1)
print("x2 =") 
print(x2)

print()
print("y =") 
print(y)


# Calling `x1 + x2` creates a new tensor containing the sum of the two inputs. However, we can also use in-place operations that are applied directly on the memory of a tensor. We therefore change the values of `x2` without the chance to re-accessing the values of `x2` before the operation. An example is shown below:

# In[11]:


x1 = torch.rand(2, 3)
x2 = torch.rand(2, 3)

print("x1 (before):") 
print(x1)
print("\nx2 (before):") 
print(x2)
print('\n---')

x2.add_(x1)

print("\nx1 (after):") 
print(x1)
print("\nx2 (after):") 
print(x2)


# In-place operations are usually marked with a underscore postfix (e.g. `add_` instead of `add`).
# 
# Another common operation aims at changing the shape of a tensor. A tensor of size `(2, 3)` can be re-organized to any other shape with the same number of elements (e.g. a tensor of size `(6,)`, or `(3, 2)`, ...). In PyTorch, this operation is called `view`:

# In[12]:


x = torch.arange(6)
print(x)


# In[13]:


x = x.view(2, 3)
print(x)


# In[14]:


x = x.permute(1, 0) # Swapping dimension 0 and 1
print(x)


# Other commonly used operations include matrix multiplications, which are essential for neural networks. Quite often, we have an input vector $\mathbf{x}$, which is transformed using a learned weight matrix $\mathbf{W}$. There are multiple ways and functions to perform matrix multiplication, some of which we list below:
# 
# |   |    |
# | :--- | :--- |
# | `torch.matmul`   | Performs the matrix product over two tensors, where the specific behavior depends on the dimensions. If both inputs are matrices (2-dimensional tensors), it performs the standard matrix product. For higher dimensional inputs, the function supports broadcasting (for details see the [documentation](https://pytorch.org/docs/stable/generated/torch.matmul.html?highlight=matmul#torch.matmul)). Can also be written as `a @ b`, similar to Numpy.   |
# | `torch.mm` | Performs the matrix product over two matrices, but doesn't support broadcasting (see [documentation](https://pytorch.org/docs/stable/generated/torch.mm.html?highlight=torch%20mm#torch.mm)). |
# | `torch.bmm` | Performs the matrix product with a support batch dimension. If the first tensor $\mathbf T$ is of shape ($b\times n\times m$), and the second tensor $\mathbf R$ ($b\times m\times p$), the output $\mathbf O$ is of shape ($b\times n\times p$), and has been calculated by performing $b$ matrix multiplications of the submatrices of $\mathbf T$ and $\mathbf R$ which can be written as $O_{ijk} = \sum_q T_{ijq} R_{iqk}.$ |
# | `torch.einsum` | Performs matrix multiplications and more (i.e. sums of products) using the Einstein summation convention. See  {ref}`below <ref/einstein_summation>`. |
# 
# Usually, we use `torch.matmul` or `torch.bmm`. We can try a matrix multiplication with `torch.matmul` below.

# In[15]:


x = torch.arange(6)
x = x.view(2, 3)
print("x =")
print(x)


# In[16]:


W = torch.arange(9).view(3, 3) # We can also stack multiple operations in a single line
print("w =") 
print(W)


# In[17]:


h = torch.matmul(x, W) # Verify the result by calculating it by hand too!
print("h =")
print(h)


# #### Indexing
# 
# We often have the situation where we need to select a part of a tensor. Indexing works just like in numpy, so let's try it:

# In[18]:


x = torch.arange(12).view(3, 4)
print("x =")
print(x)


# In[19]:


print(x[:, 1])   # Second column


# In[20]:


print(x[0])      # First row


# In[21]:


print(x[:2, -1]) # First two rows, last column


# In[22]:


print(x[1:3, :]) # Middle two rows


# ### Computational Graphs and Backpropagation
# 
# ```{margin}
# **Learnable parameters**
# ```
# 
# One of the main reasons for using PyTorch in Deep Learning projects is that we can automatically get gradients  or partial derivatives of functions that we define. We will mainly use PyTorch for implementing neural networks, and they are just fancy functions. These generally have **parameters** or **weights** whose values we want to learn.
# 
# ```{margin}
# **Computational graph**
# ```
# 
# Given an input $\mathbf{x}$, we define our function by a series of computations on that input, usually by matrix-multiplications with weight matrices and additions with so-called bias vectors. As we manipulate our input, we are automatically creating a **computational graph**. This graph shows how to arrive at our output from our input. PyTorch is a define-by-run framework; this means that we can just do our manipulations, and PyTorch will keep track of that graph for us. Thus, we create a dynamic computation graph along the way. To recap: the only thing we have to do is to compute the output, and then we can ask PyTorch to automatically get the gradients. 
# 
# ```{margin}
# **Gradient descent**
# ```
# 
# Why do we want gradients? Consider that we have defined a function, a neural net, that is supposed to compute a certain output $y$ for an input vector $\mathbf{x}$. We then define an **error measure** that tells us how wrong our network is; how bad it is in predicting output $y$ from input $\mathbf{x}$. Based on this error measure, we can use the gradients to update the weights $\mathbf{W}$ that were responsible for the output, so that the next time we present input $\mathbf{x}$ to our network, the output will be closer to what we want. The rational for the update rule is based on gradient descent, i.e. that the gradient points to the direction of greatest increase in the error function.
# 
# The first thing we have to do is to specify which tensors require gradients. By default, when we create a tensor, it does not require gradients.

# In[23]:


x = torch.ones((3,))
print(x.requires_grad)


# ```{margin}
# **Tensors = numpy arrays + GPU support + gradients**
# ```

# We can change this for an existing tensor using the function `requires_grad_()` (underscore indicating that this is an in-place operation). Alternatively, when creating a tensor, you can pass the argument `requires_grad=True` to most initializers we have seen above.

# In[24]:


x.requires_grad_(True)
print(x.requires_grad)


# ```{margin}
# **Creating a computation graph**
# ```
# 
# In order to get familiar with the concept of a computation graph, we will create one for the following function:
# 
# $$y = \frac{1}{|\mathbf{x}|}\sum_i \left[(x_i + 2)^2 + 3\right]$$
# 
# You could imagine that $\mathbf x$ are our parameters, and we want to optimize (either maximize or minimize) the output $y$. For this, we want to obtain the gradients $\partial y / \partial \mathbf{x}$. For our example, we'll use $\mathbf{x}=[0,1,2]$ as our input. This value is the reference point where we take the gradient of $y$. Here $|\cdot|$ is the length function.

# In[25]:


x = torch.arange(3, dtype=torch.float32, requires_grad=True) # Only float tensors can have gradients
print("x =", x)


# Now let's build the computation graph step by step. You can combine multiple operations in a single line, but we will separate them here to get a better understanding of how each operation is added to the computation graph.

# In[26]:


a = x + 2
b = a ** 2
c = b + 3
y = c.mean()
print("y =", y)


# ```{margin}
# **Backpropagation**
# ```
# 
# Using the statements above, we have created a computation graph that looks similar to the figure below:
# 
# <center style="width: 100%"><img src="https://uvadlc-notebooks.readthedocs.io/en/latest/_images/pytorch_computation_graph.svg" width="200px"></center>
# 
# We calculate $\mathbf a$ based on the inputs $\mathbf x$ and the constant $2$, $\mathbf{b}$ is $\mathbf{a}$ squared, and so on. The visualization is an abstraction of the dependencies between inputs and outputs of the operations we have applied.
# Each node of the computation graph has automatically defined a function for calculating the gradients with respect to its inputs, `grad_fn`. You can see this when we printed the output tensor $y$. This is why the computation graph is usually visualized in the reverse direction (arrows point from the result to the inputs). We can perform backpropagation on the computation graph by calling the function `backward()` on the last output, which effectively calculates the gradients for each tensor that has the property `requires_grad=True`:

# In[27]:


y.backward()


# `x.grad` will now contain the gradient $\partial y/ \partial \mathbf{x}$, and this gradient indicates how a change in $\mathbf{x}$ will affect output $y$ given the current input $\mathbf{x}=[0,1,2]$:

# In[28]:


print(x.grad)


# ```{margin}
# **Computing gradients by hand**
# ```
# 
# We can also verify these gradients by hand. We will calculate the gradients using the chain rule, in the same way as PyTorch did it. Note that each node depends on exactly the node above it in the graph. Moreover, there is no cross-index dependencies. Hence,
# 
# $$\frac{\partial y}{\partial x_i} = \frac{\partial y}{\partial c_i}\frac{\partial c_i}{\partial b_i}\frac{\partial b_i}{\partial a_i}\frac{\partial a_i}{\partial x_i}$$
# 
# Note that we have simplified this equation to index notation, and by using the fact that all operation besides the mean do not combine the elements in the tensor. The partial derivatives are:
# 
# $$
# \frac{\partial a_i}{\partial x_i} = 1,\hspace{1cm}
# \frac{\partial b_i}{\partial a_i} = 2\cdot a_i\hspace{1cm}
# \frac{\partial c_i}{\partial b_i} = 1\hspace{1cm}
# \frac{\partial y}{\partial c_i} = \frac{1}{3}
# $$
# 
# Hence, with the input being $\mathbf{x}=[0,1,2]$, our gradients are $\partial y/\partial \mathbf{x}=[4/3,2,8/3]$. The previous code cell should have printed the same result.

# ### GPU Support
# 
# ```{margin}
# **Introducing GPUs**
# ```
# 
# A crucial feature of PyTorch is the support of GPUs, short for Graphics Processing Unit. A GPU can perform many thousands of small operations in parallel, making it very well suitable for performing large matrix operations in neural networks. When comparing GPUs to CPUs, we can list the following main differences (credit: [Kevin Krewell, 2009](https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/)) 
# 
# <center style="width: 100%"><img src="https://uvadlc-notebooks.readthedocs.io/en/latest/_images/comparison_CPU_GPU.png" width="700px"></center>
# 
# <br>
# 
# GPUs can accelerate the training of your network up to a factor of $100$ which is essential for large neural networks. PyTorch implements a lot of functionality for supporting GPUs (mostly those of NVIDIA due to the libraries [CUDA](https://developer.nvidia.com/cuda-zone) and [cuDNN](https://developer.nvidia.com/cudnn)). First, let's check whether you have a GPU available:

# In[29]:


gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")


# ```{margin}
# **Setting the** `device` **variable**
# ```
# 
# If you have a GPU on your computer but the command above returns `False`, make sure you have the correct CUDA-version installed. By default, all tensors you create are stored on the CPU. It is often a good practice to define a `device` object in your code which points to the GPU if you have one, and otherwise to the CPU. Then, you can write your code with respect to this device object, and it allows you to run the same code on both a CPU-only system, and one with a GPU. Let's try it below. We can specify the device as follows: 

# In[30]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


# We can push a tensor to the GPU by using the function `.to(...)`, or `.cuda()`. Now let's create a tensor and push it to the device:

# In[31]:


x = torch.zeros(2, 3)
x = x.to(device)
print("x =")
print(x)


# In case you have a GPU, you should now see the attribute `device='cuda:0'` being printed next to your tensor. The zero next to cuda indicates that this is the zero-th GPU device on your computer. PyTorch also supports multi-GPU systems, but this you will only need once you have very big networks to train (if interested, see the [PyTorch documentation](https://pytorch.org/docs/stable/distributed.html#distributed-basics)). 
# 
# ```{margin}
# **GPU speedup:** matrix multiplication
# ```
# 
# We can also compare the runtime of a large matrix multiplication on the CPU with a operation on the GPU:

# In[32]:


x = torch.randn(5000, 5000)

# CPU version
start_time = time.time()
_ = torch.matmul(x, x)
end_time = time.time()
print(f"CPU time: {(end_time - start_time):6.5f}s")

# GPU version
x = x.to(device)

# CUDA is asynchronous, so we need to use different timing functions
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
_ = torch.matmul(x, x)
end.record()
torch.cuda.synchronize()  # Waits for everything to finish running on the GPU
print(f"GPU time: {0.001 * start.elapsed_time(end):6.5f}s")  # Milliseconds to seconds


# Depending on the size of the operation and the CPU / GPU in your system, the speedup of this operation can be >50x. As `matmul` operations are very common in neural networks, we can already see the great benefit of training a NN on a GPU. The time estimate can be relatively noisy here because we haven't run it for multiple times. Feel free to extend this, but it also takes longer to run.
# 
# **Reproducibility of GPU operations.** When generating random numbers, the seed between CPU and GPU is not synchronized. Hence, we need to set the seed on the GPU separately to ensure a reproducible code. Note that due to different GPU architectures, running the same code on different GPUs does not guarantee the same random numbers. Still, we don't want that our code gives us a different output every time we run it on the exact same hardware. Hence, we also set the seed on the GPU:

# In[33]:


# GPU operations have a separate seed we also want to set
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


# ## Learning the Continuous XOR
# 
# ```{margin}
# **XOR dataset with noise**
# ```
# 
# We will introduce the libraries and all additional parts you might need to train a neural network in PyTorch, using a simple example classifier on a simple yet well known example: XOR. Given two binary inputs ${x}_1$ and ${x}_2$, the label to predict is $1$ if either ${x}_1$ or ${x}_2$ is $1$ while the other is $0$, otherwise the label is $0.$ The example became famous by the fact that a single neuron, i.e. a linear classifier, cannot learn this simple function. Hence, we will learn how to build a small neural network that can learn this function. To make it a little bit more interesting, we move the XOR into continuous space and introduce some gaussian noise on the binary inputs. Our desired separation of an XOR dataset could look as follows:
# 
# <center style="width: 100%"><img src=https://uvadlc-notebooks.readthedocs.io/en/latest/_images/continuous_xor.svg width="350px"></center>

# ### Model
# 
# ```{margin}
# **Building neural nets**
# ```
# 
# If we want to build a neural network in PyTorch, we could specify all our parameters (weight matrices, bias vectors) using `Tensors` (with `requires_grad=True`), ask PyTorch to calculate the gradients and then adjust the parameters. But things can quickly get cumbersome if we have a lot of parameters. In PyTorch, there is a package called `torch.nn` that makes building neural networks more convenient. 
# 
# 
# The package `torch.nn` defines a series of useful classes like linear networks layers, activation functions, loss functions etc. A full list can be found [here](https://pytorch.org/docs/stable/nn.html). In case you need a certain network layer, check the documentation of the package first before writing the layer yourself as the package likely contains the code for it already. We import it below:

# In[34]:


import torch.nn as nn


# Additionally to `torch.nn`, there is also `torch.nn.functional`. It contains functions that are used in network layers. This is in contrast to `torch.nn` which defines them as `nn.Modules` (more on it below), and `torch.nn` actually uses a lot of functionalities from `torch.nn.functional`. Hence, the functional package is useful in many situations, and so we import it as well here.

# In[35]:


import torch.nn.functional as F


# #### nn.Module
# 
# In PyTorch, a neural network is built up out of modules. Modules can contain other modules, and a neural network is considered to be a module itself as well. The basic template of a module is as follows:

# In[36]:


class MyModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        # Some init for my module
        
    def forward(self, x):
        # Function for performing the calculation of the module.
        pass


# The `forward` function is where the computation of the module is taken place, and is executed when you call the module (`net = MyModule(); net(x)`). In the `init` function, we usually create the parameters of the module, using `nn.Parameter`, or defining other modules that are used in the forward function. The backward calculation is done automatically, but could be overwritten as well if wanted.
# 
# #### Simple classifier
# 
# We can now make use of the pre-defined modules in the `torch.nn` package, and define our own small neural network. We will use a minimal network with a input layer, one hidden layer with tanh as activation function, and a output layer. In other words, our networks should look something like this:
# 
# <center width="100%"><img src=https://uvadlc-notebooks.readthedocs.io/en/latest/_images/small_neural_network.svg width="300px"></center>
# 
# The input neurons are shown in blue, which represent the coordinates ${x}_1$ and ${x}_2$ of a data point. The hidden neurons including a tanh activation are shown in white, and the output neuron in red. The network essentially embeds each input to a subspace of $\mathbb{R}^4$ and performs logistic regression on the resulting 4-dimensional representation. In PyTorch, we can implement this as follows:

# In[37]:


class SimpleClassifier(nn.Module):

    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x


# For the examples in this notebook, we will use a tiny neural network with two input neurons and four hidden neurons. As we perform binary classification, we will use a single output neuron. Note that we do not apply a sigmoid on the output yet. This is because other functions, especially the loss, are more efficient and precise to calculate on the original outputs instead of the sigmoid output. We will discuss the detailed reason later.

# In[38]:


model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)

# Printing a module shows all its submodules
print(model)


# Printing the model lists all submodules it contains. The parameters of a module can be obtained by using its `parameters()` functions, or `named_parameters()` to get a name to each parameter object. For our small neural network, we have the following parameters:

# In[39]:


for name, param in model.named_parameters():
    print(f"Parameter {name}, shape {param.shape}")


# Each linear layer has a weight matrix of the shape `[output, input]`, and a bias of the shape `[output]`. The tanh activation function does not have any parameters. Note that parameters are only registered for `nn.Module` objects that are direct object attributes, i.e. `self.a = ...`. If you define a list of modules, the parameters of those are not registered for the outer module and can cause some issues when you try to optimize your module. There are alternatives, like `nn.ModuleList`, `nn.ModuleDict` and `nn.Sequential`, that allow you to have different data structures of modules. We will use them in a few later tutorials and explain them there. 

# ### Data
# 
# PyTorch also provides a few functionalities to load the training and test data efficiently, summarized in the package `torch.utils.data`.

# In[40]:


import torch.utils.data as data


# The data package defines two classes which are the standard interface for handling data in PyTorch: `data.Dataset`, and `data.DataLoader`. The dataset class provides an uniform interface to access the training/test data, while the data loader makes sure to efficiently load and stack the data points from the dataset into batches during training.

# #### The `Dataset` class
# 
# The dataset class summarizes the basic functionality of a dataset in a natural way. To define a dataset in PyTorch, we simply specify two functions: (1) `__getitem__`, and (2) `__len__`. The get-item function has to return the $i$-th data point in the dataset, while the len function returns the size of the dataset. For the XOR dataset, we can define the dataset class as follows:

# In[41]:


class XORDataset(data.Dataset):

    def __init__(self, size, std=0.1):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only if x is not equal to y, otherwise zero.
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, i):
        
        # Return the i-th data point of the dataset
        return self.data[i], self.label[i]


# Let's try to create such a dataset and inspect it:

# In[42]:


dataset = XORDataset(size=200)
print("Size of dataset:", len(dataset))
print("Data point 0:", dataset[0])


# To better relate to the dataset, we visualize the samples below. 

# In[43]:


def visualize_samples(data, label):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    
    plt.figure(figsize=(7, 7))
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()


# In[44]:


visualize_samples(dataset.data, dataset.label)


# #### The `DataLoader` class
# 
# The class `torch.utils.data.DataLoader` represents a Python iterable over a dataset with support for automatic batching, multi-process data loading and many more features. The data loader communicates with the dataset using the function `__getitem__`, and stacks its outputs as tensors over the first dimension to form a batch.
# In contrast to the dataset class, we usually don't have to define our own data loader class, but can just create an object of it with the dataset as input. Additionally, we can configure our data loader with the following input arguments (only a selection, see full list [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)):
# 
# | | |
# | :--- | :--- |
# | `batch_size` | Number of samples to stack per batch. |
# | `shuffle` | If True, the data is returned in a random order. This is important during training for introducing stochasticity.  |
# | `num_workers` | Number of subprocesses to use for data loading. The default, 0, means that the data will be loaded in the main process which can slow down training for datasets where loading a data point takes a considerable amount of time (e.g. large images). More workers are recommended for those, but can cause issues on Windows computers. For tiny datasets as ours, 0 workers are usually faster. |
# | `pin_memory` | If True, the data loader will copy Tensors into CUDA pinned memory before returning them. This can save some time for large data points on GPUs. Usually a good practice to use for a training set, but not necessarily for validation and test to save memory on the GPU. |
# | `drop_last` | If True, the last batch is dropped in case it is smaller than the specified batch size. This occurs when the dataset size is not a multiple of the batch size. Only potentially helpful during training to keep a consistent batch size. |
# 
# 
# Let's create a simple data loader below:

# In[45]:


data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)


# In[46]:


# next(iter(...)) catches the first batch of the data loader
# If shuffle is True, this will return a different batch every time we run this cell
# For iterating over the whole dataset, we can simple use "for batch in data_loader: ..."
data_inputs, data_labels = next(iter(data_loader))

# The shape of the outputs are [batch_size, d_1,...,d_N] where d_1,...,d_N are the 
# dimensions of the data point returned from the dataset class
print("Data inputs", data_inputs.shape, "\n", data_inputs)
print("\nData labels", data_labels.shape, "\n", data_labels)


# ### Optimization
# 
# ```{margin}
# **Training algorithm for NNs**
# ```
# 
# After defining the model and the dataset, it is time to prepare the optimization of the model. During training, we will perform the following steps:
# 
# 1. Get a batch from the data loader.
# 2. Obtain the predictions from the model for the batch.
# 3. Calculate the loss based on the difference between predictions and labels.
# 4. Backpropagation: calculate the gradients for every parameter with respect to the loss.
# 5. Update the parameters of the model in the direction of the gradients.
# 
# We have seen how we can do step 1, 2 and 4 in PyTorch. Now, we will look at step 3 and 5.

# #### Loss modules
# 
# We can calculate the loss for a batch by simply performing a few tensor operations as those are automatically added to the computation graph. For instance, for binary classification, we can use **Binary Cross Entropy** (BCE) which is defined as follows:
# 
# $$\mathcal{L}_\text{BCE} = -\sum_i \left[ y_i \log s_i + (1 - y_i) \log (1 - s_i) \right]$$
# 
# where $y$ are our labels, and $s$ our predictions, both in the range of $[0,1]$. However, PyTorch already provides a [list of predefined loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) which we can use. For instance, for BCE, PyTorch has two modules: `nn.BCELoss()`, `nn.BCEWithLogitsLoss()`. While `nn.BCELoss` expects the inputs $s$ to be in the range $[0,1]$, i.e. the output of a sigmoid, `nn.BCEWithLogitsLoss` combines a sigmoid layer and the BCE loss in a single class. This version is numerically more stable than using a plain Sigmoid followed by a BCE loss because of the logarithms applied in the loss function. Hence, it is adviced to use loss functions applied on **logits** (i.e. pre-sigmoid scores in $\mathbb{R}$) where possible. For our model defined above, we therefore use the module `nn.BCEWithLogitsLoss`. 

# In[47]:


loss_module = nn.BCEWithLogitsLoss()


# Remember to not apply a sigmoid on the output of the model in this case!

# #### Stochastic Gradient Descent
# 
# For updating the parameters, PyTorch provides the package `torch.optim` that has most popular optimizers implemented. We will discuss the specific optimizers and their differences later in the course, but will for now use the simplest of them: `torch.optim.SGD`. Stochastic Gradient Descent updates parameters by multiplying the gradients with a small constant, called **learning rate**, and subtracting those from the parameters (hence minimizing the loss). Therefore, we slowly move towards the direction of minimizing the loss. A good default value of the learning rate for a small network as ours is 0.1. 

# In[48]:


# Input to the optimizer are the parameters of the model: model.parameters()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# ```{margin}
# **Clearing gradients**
# ```
# 
# The optimizer provides two useful functions: `optimizer.step()`, and `optimizer.zero_grad()`. The step function updates the parameters based on the gradients as explained above. The function `optimizer.zero_grad()` sets the gradients of all parameters to zero. While this function seems less relevant at first, it is a crucial pre-step before performing backpropagation. If we would call the `backward` function on the loss while the parameter gradients are non-zero from the previous batch, the new gradients would actually be *added* to the previous ones instead of overwriting them. This is done because a parameter might occur multiple times in a computation graph, and we need to sum the gradients in this case instead of replacing them. Hence, remember to call `optimizer.zero_grad()` before calculating the gradients of a batch.

# ### Training
# 
# Finally, we are ready to train our model. As a first step, we create a slightly larger dataset and specify a data loader with a larger batch size. 

# In[49]:


train_dataset = XORDataset(size=2500)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)


# Now, we can write a small training function. Remember our five steps: load a batch, obtain the predictions, calculate the loss, backpropagate, and update. Additionally, we have to push all data and model parameters to the device of our choice (GPU if available). For the tiny neural network we have, communicating the data to the GPU actually takes much more time than we could save from running the operation on GPU. For large networks, the communication time is significantly smaller than the actual runtime making a GPU crucial in these cases. Still, to practice, we will push the data to GPU here. 

# In[50]:


# Push model to device. Has to be only done once
model.to(device)


# In addition, we set our model to **training mode**. This is done by calling `model.train()`. There exist certain modules that need to perform a different forward step during training than during testing (e.g. BatchNorm and Dropout), and we can switch between them using `model.train()` and `model.eval()`.

# In[51]:


def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    
    # Set model to train mode
    model.train() 
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        for data_inputs, data_labels in data_loader:
            
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output shape is [Batch size, 1], but we want [Batch size]
            
            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())
            
            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero. 
            # Otherwise, gradients would not be overwritten, but added to the existing ones.
            optimizer.zero_grad() 
            loss.backward()
            
            ## Step 5: Update the parameters
            optimizer.step()


# In[52]:


train_model(model, optimizer, train_data_loader, loss_module)


# #### Saving a model
# 
# After finish training a model, we save the model to disk so that we can load the same weights at a later time. For this, we extract the so-called `state_dict` from the model which contains all learnable parameters. For our simple model, the state dict contains the following entries:

# In[53]:


state_dict = model.state_dict()
print(state_dict)


# To save the state dictionary, we can use `torch.save`:

# In[54]:


# torch.save(object, filename). For the filename, any extension can be used
torch.save(state_dict, "our_model.tar")


# To load a model from a state dict, we use the function `torch.load` to load the state dict from the disk, and the module function `load_state_dict` to overwrite our parameters with the new values:

# In[55]:


# Load state dict from the disk (make sure it is the same name as above)
state_dict = torch.load("our_model.tar")

# Create a new model and load the state
new_model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
new_model.load_state_dict(state_dict)

# Verify that the parameters are the same
print("Original model\n", model.state_dict())
print("\nLoaded model\n", new_model.state_dict())


# A detailed tutorial on saving and loading models in PyTorch can be found [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

# ### Evaluation
# 
# Once we have trained a model, it is time to evaluate it on a held-out test set. As metric we will use accuracy since the dataset is balanced. As our dataset consist of randomly generated data points, we need to first create a test set with a corresponding data loader.

# In[56]:


test_dataset = XORDataset(size=500)
test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False) 


# When evaluating the model, we don't need to keep track of the computation graph as we don't intend to calculate the gradients. This reduces the required memory and speed up the model. In PyTorch, we can deactivate the computation graph using the `with torch.no_grad(): ...` context manager. Remember to additionally set the model to **eval mode**.

# In[57]:


def eval_model(model, data_loader):
    model.eval() # Set model to eval mode
    true_preds, num_preds = 0., 0.
    
    with torch.no_grad(): # Context manager: deactivate gradients for the ff. code
        for data_inputs, data_labels in data_loader:
            
            # Determine prediction of model on dev set
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)
            preds = torch.sigmoid(preds) # Sigmoid to map predictions between 0 and 1
            pred_labels = (preds >= 0.5).long() # Binarize predictions to 0 and 1
            
            # Keep records of predictions for the accuracy metric (true_preds=TP+TN, num_preds=TP+TN+FP+FN)
            true_preds += (pred_labels == data_labels).sum()
            num_preds += data_labels.shape[0]
            
    acc = true_preds / num_preds
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")


# If we trained our model correctly, we should see a score close to 100% accuracy. However, this is only possible because of our simple task, and unfortunately, we usually don't get such high scores on test sets of more complex tasks.

# In[58]:


eval_model(model, test_data_loader)


# #### Visualizing classification boundaries
# 
# To visualize what our model has learned, we can perform a prediction for every data point in a range of $[-0.5, 1.5]$, and visualize the predicted class as in the sample figure at the beginning of this section. This shows where the model has created decision boundaries, and which points would be classified as $0$, and which as $1$. We therefore get a background image out of blue (class 0) and orange (class 1). The spots where the model is uncertain we will see a blurry overlap. The specific code is less relevant compared to the output figure which should hopefully show us a clear separation of classes:

# In[59]:


@torch.no_grad() # Decorator, same effect as "with torch.no_grad(): ..." over the whole function. (!)
def visualize_classification(model, data, label, figsize=(6, 6)):
    
    ## Plot data points
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    
    data_0 = data[label == 0]
    data_1 = data[label == 1]
    
    fig = plt.figure(figsize=figsize)
    plt.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()
    
    ## Plot decision function as color gradient
    model.to(device)

    c0 = torch.Tensor(to_rgba("C0")).to(device)
    c1 = torch.Tensor(to_rgba("C1")).to(device)
    x1 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    x2 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds = model(model_inputs)
    preds = torch.sigmoid(preds)
    
    output_image = (1 - preds) * c0[None, None] + preds * c1[None, None]  # Specifying "None" in a dimension creates a new one
    output_image = output_image.cpu().numpy()  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    
    return fig


_ = visualize_classification(model, dataset.data, dataset.label, figsize=(7, 7))
plt.show()


# The decision boundaries might not look exactly as in the figure in the preamble of this section which can be caused by running it on CPU or a different GPU architecture. Nevertheless, the result on the accuracy metric should be the approximately the same. 

# ## TensorBoard Logging
# 
# TensorBoard is a logging and visualization tool that is a popular choice for training deep learning models. Although initially published for TensorFlow, TensorBoard is also integrated in PyTorch allowing us to easily use it. First, let's import it below.

# In[60]:


from torch.utils.tensorboard import SummaryWriter
get_ipython().run_line_magic('load_ext', 'tensorboard')


# The last line is required if you want to run TensorBoard directly in the Jupyter Notebook. Otherwise, you can start TensorBoard from the terminal.
# 
# PyTorch's TensorBoard API is simple to use. We start the logging process by creating a new object, `writer = SummaryWriter(...)`, where we specify the directory in which the logging file should be saved. With this object, we can log different aspects of our model by calling functions of the style `writer.add_...`. For example, we can visualize the computation graph with the function `writer.add_graph`, or add a scalar value like the loss with `writer.add_scalar`. Let's adapt our initial training function with adding a TensorBoard logger below.

# In[61]:


def train_model_with_logger(model, optimizer, data_loader, loss_module, val_dataset, num_epochs=100, logging_dir='runs/our_experiment'):
    # Create TensorBoard logger
    writer = SummaryWriter(logging_dir)
    model_plotted = False
    
    # Set model to train mode
    model.train() 
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for data_inputs, data_labels in data_loader:
            
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            
            # For the very first batch, we visualize the computation graph in TensorBoard
            if not model_plotted:
                writer.add_graph(model, data_inputs)
                model_plotted = True
            
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], but we want [Batch size]
            
            ## Step 3: Calculate the loss
            loss = loss_module(preds, data_labels.float())
            
            ## Step 4: Perform backpropagation
            optimizer.zero_grad() 
            loss.backward()
            
            ## Step 5: Update the parameters
            optimizer.step()
            
            ## Step 6: Take the running average of the loss
            epoch_loss += loss.item()
            
        # Add average loss to TensorBoard
        epoch_loss /= len(data_loader)
        writer.add_scalar('training_loss',
                          epoch_loss,
                          global_step = epoch + 1)
        
        # Visualize prediction and add figure to TensorBoard
        # Since matplotlib figures can be slow in rendering, we only do it every 10th epoch
        if (epoch + 1) % 10 == 0:
            fig = visualize_classification(model, val_dataset.data, val_dataset.label)
            writer.add_figure('predictions',
                              fig,
                              global_step = epoch + 1)
    
    writer.close()


# Let's use this method to train a model as before, with a new model and optimizer.

# In[62]:


model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
train_model_with_logger(model, optimizer, train_data_loader, loss_module, val_dataset=dataset)


# The TensorBoard file in the folder `runs/our_experiment` now contains a loss curve, the computation graph of our network, and a visualization of the learned predictions over number of epochs. To start the TensorBoard visualizer, simply run the following statement:

# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir runs/our_experiment')


# <center><img src=https://uvadlc-notebooks.readthedocs.io/en/latest/_images/tensorboard_screenshot1.png width="1100px"></center>
# 
# TensorBoard visualizations can help to identify possible issues with your model, and identify situations such as overfitting. You can also track the training progress while a model is training, since the logger automatically writes everything added to it to the logging file. Feel free to explore the TensorBoard functionalities, and we will make use of TensorBoards a couple of times from Tutorial 5 on.

# (ref/einstein_summation)=
# ## Appendix: Einstein summation

# <b>Rules</b>
# 
# 1. Repeated indices are summed over.
# 2. Omitting an index means that axis will be summed.
# 3. Unsummed axes can be returned in any order.
# 

# In[64]:


# Summing all entries of a tensor from Rule 2 -- einsum also implemented in numpy!
print(np.einsum('i ->', np.ones(3)))
print(torch.einsum('i ->', torch.ones(3)))


# Einstein summation lets us perform sums over products without having to do reshape gymnastics:

# In[65]:


A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

B = torch.tensor([
    [4, 1],
    [5, 0],
    [1, 1]
])

# Matrix-matrix multiplication
torch.einsum('ij, jk -> ik', A, B)


# In[66]:


# Matrix-vector multiplication
v = torch.tensor([1, 1, 1])
torch.einsum('ij, j -> i', A, v)


# In[67]:


# Sum all rows, i.e. we are left with column index
torch.einsum('ij -> j', A)


# In[68]:


# Hadamard product
torch.einsum('ij, ij -> ij', A, A)


# In[69]:


# Outer product
torch.einsum('i, j -> ij', v, v)


# In[70]:


# Get diagonal elements of a matrix
torch.einsum('ii -> i', torch.eye(3))


# In[71]:


# Compute trace
torch.einsum('ii ->', torch.eye(3))

