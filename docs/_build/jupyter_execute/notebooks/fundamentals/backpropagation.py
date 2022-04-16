#!/usr/bin/env python
# coding: utf-8

# # Backpropagation on DAGs

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

# In this notebook, we will look at **backpropagation** (BP) on **directed acyclic computational graphs** (DAG). Our main result is that a single training step for a single data point (consisting of both a forward and a backward pass) has a time complexity that is linear in the number of edges of the network. In the last section, we take a closer look at the implementation of `.backward` in PyTorch.
# 
# **Readings**
# * [Evaluating $\nabla f(x)$ is as fast as $f(x)$](https://timvieira.github.io/blog/post/2016/09/25/evaluating-fx-is-as-fast-as-fx/)
# * [Back-propagation, an introduction](http://www.offconvex.org/2016/12/20/backprop/)

# ## Gradient descent on the loss surface

# ```{margin}
# **Constructing the loss surface**. The loss function $\ell$ acts as an almost-everywhere differentiable surrogate to the true objective. The empirical loss surface will generally vary for different samples drawn. But we except these surfaces to be very similar, assuming the samples are drawn from the same distribution.
# ```
# 
# For every data point $(\mathbf x, y)$, the loss function $\ell$ assigns a nonnegative number $\ell(y, f_{\mathbf w}(\mathbf x))$ that approaches zero whenever the predictions $f_{\mathbf w}(\mathbf x)$ approach the target values $y$. Given the current parameters $\mathbf w \in \mathbb R^d$ of a neural network $f$, we can imagine the network to be at a certain point $(\mathbf w, \mathcal L_{\mathcal X}(\mathbf w))$ on a surface in $\mathbb R^d \times \mathbb R$ where $\mathcal L_{\mathcal X}(\mathbf w)$ is the average loss over the dataset:
# 
# $$
# \mathcal L_{\mathcal X}(\mathbf w) = \frac{1}{|\mathcal X|} \sum_{(\mathbf x, y) \in \mathcal X} \ell(y, f_{\mathbf w}(\mathbf x)).
# $$
# 
# So training a neural network is equivalent to finding the minimum of this surface. In practice, we use variants of gradient descent, characterized by the update rule $\mathbf w \leftarrow \mathbf w - \varepsilon \nabla_{\mathbf w} \mathcal L_{\mathcal X}$, to find a local minimum. Here $-\nabla_{\mathbf w} \mathcal L_{\mathcal X}$ is the direction of steepest descent at $\mathbf w$ and the learning rate $\varepsilon > 0$ is a constant that controls the step size.
# 
# 
# ```{figure} ../../img/loss_surface_resnet.png
# ---
# name: loss-surface-resnet
# width: 35em
# ---
# Much of deep learning research is dedicated to studying the geometry of loss surfaces and its effect on optimization. **Source**: Visualizing the Loss Landscape of Neural Nets
#  [[arxiv.org/abs/1712.09913](https://arxiv.org/abs/1712.09913)]
# ```
# 
# 

# ```{margin}
# **Derivatives of comp. graphs**
# ```
# 
# In principle, we can perturb the current state of the network (obtained during forward pass) by perturbing the network weights / parameters. This results in perturbations flowing up to the final loss node (assuming each computation is differentiable). So it's not a mystery that we can compute derivatives of computational graphs which may appear, at first glance, as "discrete" objects. Another perspective is that a computational DAG essentially models a a sequence of function compositions which can be easily differentiated using chain rule. However, looking at the network structure allows us to easily code the computation into a computer, exploit modularity, and efficiently compute the flow of derivatives at each layer. This is further discussed below.
# 
# 
# ```{margin}
# **The need for efficient BP**
# ```
# 
# Observe that $\nabla_\mathbf w \mathcal L_{\mathcal X}$ consists of partial derivatives for each weight in the network. This can easily number in millions. So this backward pass operation can be huge. To compute these values efficiently, we will perform both forward and backward passes in a dynamic programming fashion to avoid recomputing any known value. As an aside, this improvement in time complexity turns out to be insufficient for pratical uses, and is supplemented with sophisticated hardware for parallel computation (GPUs / TPUs) which can reduce training time by some factor, e.g. from days to hours.

# ## Backpropagation on Computational Graphs

# ```{margin}
# **Forward pass** 
# ```
# 
# A neural network can be modelled as a **directed acyclic graph** (DAG) of compute and parameter nodes that implements a function $f$ and can be extended to implement the calculation of the loss value for each training example and parameter values. In computing $f(\mathbf x)$, the input $\mathbf x$ is passed to the first layer and propagated forward through the network, computing the output value of each node. Every value in the nodes is stored to preserve the current state for backward pass, as well as to avoid recomputation for the nodes in the next layer. Assuming a node with $n$ inputs require $n$ operations, then one forward pass takes $O(E)$ calculations were $E$ is the number of edges of the graph.

# ```{figure} ../../img/backprop-compgraph2.png
# ---
# width: 35em
# name: backprop-compgraph2
# ---
# Backpropagation through a single layer neural network with weights $w_0$ and $w_1$, and input-output pair $(x, y).$ Shown here is the gradient flowing from the loss node $\mathcal L$ to the weight $w_0.$ [[source]](https://drive.google.com/file/d/1JCWTApGieKZmFW4RjCANZM8J6igcsdYg/view)
# ```

# ```{margin}
# **Backward pass** 
# ```
# 
# During backward pass, we divide gradients into two groups: **local gradients** ${\frac{\partial{\mathcal u}}{\partial w}}$ between connected nodes $u$ and $w,$ and **backpropagated gradients** ${\frac{\partial{\mathcal L}}{\partial u}}$ for each node ${u}.$ Our goal is to calculate the backpropagated gradient of the loss with respect to parameter nodes. Note that parameter nodes have zero fan-in ({numref}`backprop-compgraph2`). BP proceeds inductively. First, $\frac{\partial{\mathcal L}}{\partial \mathcal L} = 1$ is stored as gradient of the node which computes the loss value. If the backpropagated gradient ${\frac{\partial{\mathcal L}}{\partial u}}$ is stored for each compute node $u$ in the upper layer, then after computing local gradients ${\frac{\partial{u}}{\partial w}}$, the backpropagated gradient ${\frac{\partial{\mathcal L}}{\partial w}}$ for compute node $w$ can be calculated via the chain rule:
# 
# ```{math}
# :label: backprop
# {\frac{\partial\mathcal L}{\partial w} } = \sum_{ {u} }\left( {{\frac{\partial\mathcal L}{\partial u}}} \right)\left( {{\frac{\partial{u}}{\partial w}}} \right).
# ```

# ```{margin}
# BP is a useful tool for understanding how derivatives flow through a model. This can be extremely helpful in reasoning about why some models are difficult to optimize. Classic examples are vanishing or exploding gradients as we go into deeper layers of the network.
# ```
# 
# Thus, continuing the "flow" of gradients to the current layer. The process ends on  nodes with zero fan-in. Note that the partial derivatives are evaluated on the current network state &mdash; these values are stored during forward pass which precedes backward pass. Analogously, all backpropagated gradients are stored in each compute node for use by the next layer. On the other hand, there is no need to store local gradients; these are computed as needed. Hence, it suffices to compute all gradients with respect to compute nodes to get all gradients with respect to the weights of the network.
#     
# 
# ```{figure} ../../img/backprop-compgraph.png
# ---
# width: 30em
# name: backprop-compgraph
# ---
# BP on a generic comp. graph with fan out > 1 on node <code>y</code>. Each backpropagated gradient computation is stored in the corresponding node. For node <code>y</code> to calculate the backpropagated gradient we have to sum over the two incoming gradients which can be implemented using matrix multiplication of the gradient vectors. [[source]](https://drive.google.com/file/d/1JCWTApGieKZmFW4RjCANZM8J6igcsdYg/view)
# ```
# 

# **Backpropagation algorithm.** Now that we know how to compute each backpropagated gradient implemented as `u.backward()` for node `u` which sends its gradient $\frac{\partial \mathcal L}{\partial u}$ to all its parent nodes (nodes on the lower layer connected to `u`). The complete recursive algorithm with SGD is implemented below. Note that this abstracts away autodifferentiation.
# 
# ```python 
# def Forward():
#     for c in compute: 
#         c.forward()
# 
# def Backward(loss):
#     for c in compute: c.grad = 0
#     for c in params:  c.grad = 0
#     for c in inputs:  c.grad = 0
#     loss.grad = 1
# 
#     for c in compute[::-1]: 
#         c.backward()
# 
# def SGD(eta):
#     for w in params:
#         w.value -= eta * w.grad
# ```

# <br>

# Two important properties of the algorithm which makes it the practical choice for training huge neural networks are as follows:

# * **Modularity.** The dependence only on nodes belonging to the upper layer suggests a modularity in the computation, e.g. we can connect DAG subnetworks with possibly distinct network architectures by only connecting nodes that are exposed between layers. 
# 
# <br>
# 
# * **Efficiency.** From the backpropagation equation {eq}`backprop`, the backpropagated gradient $\frac{\partial \mathcal L}{\partial w}$ for node $w$ is computed by a sum that is indexed by $u$ for every node connected to $w$. Iterating over all nodes $w$ in the network, we cover all the edges in the network with no edge counted twice. Assuming computing local gradients take constant time, then backward pass requires $O(E)$ computations.

# $\phantom{3}$

# ```{admonition} BP equations for MLPs
# 
# Consider an MLP which can be modelled as a computational DAG with edges between preactivation and activation values, as well as edges from weights and input values that fan into preactivations ({numref}`backprop-compgraph2`). Let ${z_j}^{[t]} = \sum_k {w_{jk}}^{[t]}{a_k}^{[t-1]}$ and ${\mathbf a}^{[t]} = \phi^{[t]}({\mathbf z}^{[t]})$ be the values of compute nodes at the $t$-th layer of the network. The backpropagated gradients for the compute nodes of the current layer are given by
#     
# $$\begin{aligned}
#         \dfrac{\partial \mathcal L}{\partial {a_j}^{[t]}} 
#         &= \sum_{k}\dfrac{\partial \mathcal L}{\partial {z_k}^{[t+1]}} \dfrac{\partial {z_k}^{[t+1]}}{\partial {a_j}^{[t]}} = \sum_{k}\dfrac{\partial \mathcal L}{\partial {z_k}^{[t+1]}} {w_{kj}}^{[t+1]}
#     \end{aligned}$$
# 
# and
# 
# $$\begin{aligned}
#     \dfrac{\partial \mathcal L}{\partial {z_j}^{[t]}} 
#     &= \sum_{l}\dfrac{\partial \mathcal L}{\partial {a_l}^{[t]}} \dfrac{\partial {a_l}^{[t]}}{\partial {z_j}^{[t]}}.
# \end{aligned}$$
# 
# This sum typically reduces to a single term for activations such as ReLU, but not for activations which depend on multiple preactivations such as softmax. Similarly, the backpropagated gradients for the parameter nodes (weights and biases) are given by
# 
# $$\begin{aligned}
#     \dfrac{\partial \mathcal L}{\partial {w_{jk}}^{[t]}} 
#     &= \dfrac{\partial \mathcal L}{\partial {z_j}^{[t]}} \dfrac{\partial {z_j}^{[t]}}{\partial {w_{jk}}^{[t]}} = \dfrac{\partial \mathcal L}{\partial {z_j}^{[t]}} {a^{[t-1]}_k} \\
#     \text{and}\qquad\dfrac{\partial \mathcal L}{\partial {b_{j}}^{[t]}} 
#     &= \dfrac{\partial \mathcal L}{\partial {z_j}^{[t]}} \dfrac{\partial {z_j}^{[t]}}{\partial {b_{j}}^{[t]}} = \dfrac{\partial \mathcal L}{\partial {z_j}^{[t]}}.
# \end{aligned}$$
# 
# Backpropagated gradients for compute nodes are stored until the weights are updated, e.g. $\frac{\partial \mathcal L}{\partial {z_k}^{[t+1]}}$ are retrieved in the compute nodes of the $t+1$-layer to compute gradients in the $t$-layer. On the other hand, the local gradients $\frac{\partial {a_k}^{[t]}}{\partial {z_j}^{[t]}}$ are computed directly using autodifferentiation and evaluated with the current network state obtained during forward pass.
# ```

# ## Autodifferentiation with PyTorch `autograd`
# 
# The `autograd` package allows automatic differentiation by building computational graphs on the fly every time we pass data through our model. Autograd tracks which data combined through which operations to produce the output. This allows us to take derivatives over ordinary imperative code. This functionality is consistent with the memory and time requirements outlined in above for BP.
# 
# <br>
# 
# **Backward for scalars.** Let $y = \mathbf x^\top \mathbf x = \sum_i {x_i}^2.$ In this example, we initialize a tensor `x` which initially has no gradient. Calling backward on `y` results in gradients being stored on the leaf tensor `x`. 

# In[132]:


x = torch.arange(4, dtype=torch.float, requires_grad=True)
y = x.T @ x 

y.backward() 
(x.grad == 2*x).all()


# **Backward for vectors.** Let $\mathbf y = g(\mathbf x)$ and let $\mathbf v$ be a vector having the same length as $\mathbf y.$ Then `y.backward(v)` implements   
# 
# $$\sum_i v_i \left(\frac{\partial y_i}{\partial x_j}\right)$$ 
#   
# resulting in a vector of same length as `x` that is stored in `x.grad`. Note that the terms on the right are the local gradients in backprop. Hence, if `v` contains backpropagated gradients of nodes that depend on `y`, then this operation gives us the backpropagated gradients with respect to `x`, i.e. setting $v_i = \frac{\partial \mathcal{L} }{\partial y_i}$ gives us the vector $\frac{\partial \mathcal{L} }{\partial x_j}.$

# In[179]:


x = torch.rand(size=(4,), dtype=torch.float, requires_grad=True)
v = torch.rand(size=(2,), dtype=torch.float)
y = x[:2]

# Computing the Jacobian by hand
J = torch.tensor(
    [[1, 0, 0, 0],
    [0, 1, 0, 0]], dtype=torch.float
)

# Confirming the above formula
y.backward(v)
(x.grad == v @ J).all()


# **Locally disabling gradient tracking.** Disabling gradient computation is useful when computing values, e.g. accuracy, whose gradients will not be backpropagated into the network. To stop PyTorch from building computational graphs, we can put the code inside a `torch.no_grad()` context or inside a function with a `@torch.no_grad()` decorator.
# 
# Another technique is to use the `.detach()` method which returns a new tensor detached from the current graph but shares the same storage with the original one. In-place modifications on either of them will be seen, and may trigger errors in correctness checks.
