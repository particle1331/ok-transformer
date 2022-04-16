#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Datasets 

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

# To train neural networks, we typically need to stream large amounts of input data  &mdash; data that is too large to fit in computer memory. In this case, we can't use `fit` on Keras models, and we have to load the data from storage in chunks as we shall see later. TensorFlow provides the `tf.data.Dataset` API to facilitate efficient input pipelines. As we will see below, `Dataset` usage follows a common pattern:
# 
# 1. Create a source dataset from your input data.
# 2. Apply dataset transformations to preprocess the data.
# 3. Iterate over the dataset and process the elements.

# ```{margin}
# ⚠️ **Attribution:** This notebook builds on top of [these notebooks](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/ch13/) of {cite}`RaschkaMirjalili2019`. These notebooks are released under [MIT License](https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/LICENSE.txt). 
# ```

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt 
import pandas as pd

from inefficient_networks.config import config
from inefficient_networks import utils

config.set_matplotlib() 
config.set_ignore_warnings()
config.set_tensorflow_seeds(1)
print(config.list_tensorflow_devices())
print(tf.__version__)


# ## Dataset from tensors

# We can initialize a TF dataset from an existing tensor as follows:

# In[2]:


a = tf.range(10)
ds = tf.data.Dataset.from_tensor_slices(a)
print(ds)


# In[3]:


for item in ds:
    print(item)


# In[4]:


ds_batch = ds.batch(4, drop_remainder=False) # Analogous to drop_last in PyTorch
for batch in ds_batch:
    print(batch)


# To create joint datasets, simply pass a tuple of tensors in `from_tensor_slices`:

# In[5]:


X = tf.random.uniform([10, 3], dtype=tf.float32)
y = tf.range(10)

ds_joint = tf.data.Dataset.from_tensor_slices((X, y))
for x, t in ds_joint.batch(4):
    print(pd.DataFrame({
        'X1': x.numpy()[:, 0],
        'X2': x.numpy()[:, 1], 
        'X3': x.numpy()[:, 2], 
        'y': t.numpy()
    }), '\n')


# ## Transformations

# Applying transformations to each individual element of a TF dataset is easy &mdash; just call `map`. This will return a dataset where each streamed instance is a transformed version of the original instance.

# In[6]:


X_max = tf.reduce_max(X, axis=0)
X_min = tf.reduce_min(X, axis=0)
ds_transformed = ds_joint.map(lambda x, y: (tf.math.divide(x - X_min, X_max - X_min), y))
for x, t in ds_transformed.batch(4):
    print(pd.DataFrame(
            {
                'X1': x.numpy()[:, 0],
                'X2': x.numpy()[:, 1], 
                'X3': x.numpy()[:, 2], 
                'y': t.numpy()
            }
        ), '\n'
    )


# Applying this sort of transformation can be used for a user-defined function. 
# For example, if we have a dataset created from the list of image filenames on disk, 
# we can define a function to load the images from these filenames and apply that 
# function by calling the `.map()` method. 

# ## Creating epochs

# To train a neural network using SGD, 
# it is important to feed training data as randomly shuffled batches. (Otherwise, it biases the weight updates with the ordering of the input data.) TF provides a `.shuffle` method on dataset objects with a `buffer_size` parameter. This [answer](https://stackoverflow.com/a/47025850) in SO provides a good explanation for `buffer_size`:
# 
# > [`Dataset.shuffle()` is designed] to handle datasets that are too large to fit in memory. Instead of shuffling the entire dataset, it maintains a buffer of `buffer_size` elements, and randomly selects the next element from that buffer (replacing it with the next input element, if one is available). <br><br>
# Changing the value of `buffer_size` affects how uniform the shuffling is: if `buffer_size` is greater than the number of elements in the dataset, you get a uniform shuffle; if it is 1 then you get no shuffling at all. For very large datasets, a typical "good enough" approach is to randomly shard the data into multiple files once before training, then shuffle the filenames uniformly, and then use a smaller shuffle buffer. However, the appropriate choice will depend on the exact nature of your training job.

# In[7]:


fig = plt.figure(figsize=(10, 4))
buffer_size = [1, 20, 60, 100]
for i in range(len(buffer_size)):
    shuffled_data = []
    ds_range = tf.data.Dataset.from_tensor_slices(tf.range(100))
    for x in ds_range.shuffle(buffer_size[i]).batch(1):
        shuffled_data.append(x.numpy()[0])

    ax = fig.add_subplot(1, 4, i+1)
    ax.bar(range(100), shuffled_data)
    ax.set_title(f"buffer_size={buffer_size[i]}")

plt.tight_layout()
plt.show()


# Furthermore, `shuffle` has an important argument `reshuffle_each_iteration` that controls whether the shuffle order should be different each time the dataset is iterated over. This is set to `True` by default. 

# In[8]:


dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3)
dataset = dataset.repeat(2)
list(dataset.as_numpy_iterator())


# In[9]:


dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
dataset = dataset.repeat(2)
list(dataset.as_numpy_iterator())


# ### Shuffle, batch, repeat

# When training a model for multiple epochs, we need to shuffle and iterate over the dataset by the desired number of epochs. To repeat the dataset, we use the `.repeat` method on the dataset object. The following pattern is the correct order of creating epochs. For training, it is recommended to set `drop_remainder=True` in `.batch()` to drop the last mini batch of size 1. 

# In[10]:


buffer_size = 6
for x, t in ds_transformed.shuffle(buffer_size).batch(3).repeat(2):
    print(pd.DataFrame(
            {
                'X1': x.numpy()[:, 0],
                'X2': x.numpy()[:, 1], 
                'X3': x.numpy()[:, 2], 
                'y': t.numpy()
            }
        ), '\n'
    )


# To see this more transparently, consider the 1-dimensional dataset:

# In[11]:


data = tf.data.Dataset.range(10).shuffle(6).batch(3).repeat(2)
list(data.as_numpy_iterator())


# Iterating over a dataset to generate batches can also be useful. Again the default behavior of reshuffling at each iteration turns out to be very convenient.

# In[12]:


data = tf.data.Dataset.range(10).shuffle(10).batch(3, drop_remainder=True)
num_epochs = 2
for i in range(num_epochs):
    print(f"\n[Epoch {i}]:")
    for x in data:
        print("  ", x)


# ## Dataset from local files

# We can get filenames using `.glob` on a `pathlib.Path` object as follows:

# In[13]:


utils.download_kaggle_dataset("waifuai/cat2dog")

cat_imgdir_path = config.DATASET_DIR / "cat2dog" / "cat2dog" / "trainA"
dog_imgdir_path = config.DATASET_DIR / "cat2dog" / "cat2dog" / "trainB"

cat_file_list = sorted([str(path) for path in cat_imgdir_path.glob("*.jpg")])
dog_file_list = sorted([str(path) for path in dog_imgdir_path.glob("*.jpg")])


# Visualizing image sets for cats and dogs:

# In[14]:


import pathlib

fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(cat_file_list[:6]):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(pathlib.Path(file).name, size=15)
    
plt.tight_layout()
plt.show()


# In[15]:


fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(dog_file_list[:6]):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(pathlib.Path(file).name, size=15)
    
plt.tight_layout()
plt.show()


# Instead of having a dataset of arrays for images, and their corresponding labels, we can create a dataset of filenames and their labels. Then, we can transform the filenames to images using a mapping to load and preprocess images given their filenames.

# In[16]:


from functools import partial

# Define mapping function: (filename, label) -> (RGB array, label)
def load_and_preprocess(path, label, img_width=124, img_height=124):
    img_raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img /= 255.0
    return img, label

# Create dataset of RGB arrays resized to 32x32x3
filenames = cat_file_list + dog_file_list
labels = [0] * len(cat_file_list) + [1] * len(dog_file_list)
filenames_dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
images_dataset = filenames_dataset.map(partial(load_and_preprocess, img_width=32, img_height=32))

# Display one image and its label (0 = cat, 1 = dog) 
img, label = next(iter(images_dataset.batch(1)))
print(label.numpy()[0])
plt.imshow(img[0, :, :, :]);


# ## Datasets from `tensorflow_datasets`

# The `tensorflow_datasets` library provides a collection of freely available 
# (well formatted) datasets for training or evaluating deep learning models which allows for quick 
# experimentation. The datasets also come with an `info` dictionary which contains all relevant metadata.
# Morevero, the datasets already load as a `Dataset` object. The list of all available datasets can be found in [this catalog](https://www.tensorflow.org/datasets/catalog/overview).

# In[17]:


import tensorflow_datasets as tfds

print(len(tfds.list_builders())) # no. of available datasets
print(tfds.list_builders()[:5])


# Datasets from `tfds` can be loaded using three steps:

# In[18]:


coil100_bldr = tfds.builder('coil100')                      # (1)
coil100_bldr.download_and_prepare()                         # (2)
coil100_ds = coil100_bldr.as_dataset(shuffle_files=True)    # (3)


# In[19]:


import json
json_info = json.loads(coil100_bldr.info.as_json)
print(json.dumps(json_info, indent=2))


# We can see that the result is a dictionary:

# In[20]:


print(coil100_ds)


# There is only train set in this dataset.

# In[21]:


coil100_ds_trn = coil100_ds['train']
print(coil100_ds_trn)
print(isinstance(coil100_ds_trn, tf.data.Dataset))
print(len(coil100_ds_trn))


# In[22]:


instance = next(iter(coil100_ds_trn))
print(type(instance))
print(instance.keys())


# Each element of this dataset is a dictionary, so we have to extract the features and labels using a mapping:

# In[23]:


ds_train = coil100_ds_trn.map(lambda d: (
    {k: d[k] for k in d.keys() if k != 'object_id'},
    d['object_id']
))

# Try one example
features, labels = next(iter(ds_train.batch(8)))
print(features['angle'].shape)
print(features['angle_label'].shape)
print(features['image'].shape)
print(labels.numpy().shape)


# In[24]:


fig = plt.figure(figsize=(10, 5))
for i in range(8):
    img = features['image'][i, :, :, :]
    ax = fig.add_subplot(2, 4, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(f"angle={features['angle'][i]}, id={labels[i].numpy()}", size=15)
    
plt.tight_layout()
plt.show()


# It turns out that `tfds` has a wrapper function called `load` that performs all the three steps. We will use this to fetch the MNIST dataset in one step:

# In[25]:


MNIST, MNIST_info = tfds.load('mnist', with_info=True, shuffle_files=False)
print(MNIST_info)


# In[26]:


train_dataset, test_dataset = MNIST['train'], MNIST['test']
train_dataset = train_dataset.map(lambda d: (d['image'], d['label']))
test_dataset = test_dataset.map(lambda d: (d['image'], d['label']))

print(type(train_dataset))
print(train_dataset)


# In[27]:


import matplotlib.pyplot as plt

images, labels = next(iter(train_dataset.batch(8)))
fig = plt.figure(figsize=(10, 5))
for i in range(8):
    img = images[i, :, :, :]
    ax = fig.add_subplot(2, 4, i+1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img, cmap='gray_r')
    ax.set_title(f"{labels[i].numpy()}", size=15)
    
plt.tight_layout()
plt.show()


# ## Training Keras models

# ```{margin}
# ⚠️ This section requires knowledge of the **Keras API** discussed in the notebook [Mechanics of TensorFlow](https://particle1331.github.io/inefficient-networks/notebooks/tensorflow/02-tensorflow-mechanics.html).
# ```
# 
# So far we have learned about the basic utility components of 
# TensorFlow for manipulating tensors and organizing data into formats that we 
# can iterate over during training. In this section, we look at how to feed data into TensorFlow models.

# ### Custom training loop

# **Dataset.** In this section, we train a simple linear regression model derived from `tf.keras.Model` by implementing SGD from scratch. The loop iterates over a `tf.data.Dataset` object which acts as a data loader. We use a 2-layer MLP to learn the artificial dataset composed of 10 points below.

# In[28]:


import numpy as np

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])

plt.figure(figsize=(7, 6), dpi=80)
plt.scatter(X_train, y_train, edgecolor="#333", s=50)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression dataset')
plt.show()


# In[29]:


X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
ds_train_orig = tf.data.Dataset.from_tensor_slices((
    tf.cast(X_train_norm, tf.float32), 
    tf.cast(y_train, tf.float32)
))


# <br>
# 
# **Model.** We implement a univariate linear regression model by subclassing the Keras `Model` class.

# In[30]:


class RegressionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')
    
    def call(self, x):
        return self.w * x + self.b


# <br>
# 
# **Training loop.** The `train` function implements a single step of SGD optimization where gradients of the MSE loss function obtained automatically are used to update the weight `w` and bias `b`. Note that using `count=None` on `repeat` will create a batched version of the dataset that repeats infinitely many times. But since we implement no early stopping mechanism, we set `count=200` to train the model for 200 epochs. 

# In[31]:


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

@tf.function
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        loss = loss_fn(model(inputs), outputs)
    
    dw, db = tape.gradient(loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)


# Alternatively, we can exploit the `apply_gradients` method of built-in Keras optimizers:
# 
# ```python
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
# optimizer.apply_gradients(zip([dw, db], [model.w, model.b]))
# ```

# Finally, we can implement the train loop by iterating over the batch loader and applying the train step at each iteration:

# In[32]:


# Hyperparameters
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 1

# Instantiate model
model = RegressionModel()

# Create batch loader
ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.batch(1)
ds_train = ds_train.repeat(count=NUM_EPOCHS)

ws, bs = [], []
steps_per_epoch = int(np.ceil(len(y_train) / BATCH_SIZE))
for i, batch in enumerate(ds_train):
    ws.append(model.w.numpy())
    bs.append(model.b.numpy())

    bx, by = batch
    loss_value = loss_fn(model(bx), by)
    train(model, bx, by, learning_rate=LEARNING_RATE)
    
    if i%100==0:
        print(f'Epoch {i//steps_per_epoch:4d} Loss {loss_value:>8.4f}')


# <br>
# 
# **History.** Here we plot the learned model and the history of its parameters. As training progresses, the weight and bias converge to an optimal value.

# In[33]:


print(f'Final Parameters: w={model.w.numpy():.4f}, b={model.b.numpy():.4f}')

# Generate test set; here test = inference
X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

# Get predictions on test set
y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))

# Plot learned model
fig = plt.figure(figsize=(13, 5), dpi=600)
ax = fig.add_subplot(1, 2, 1)
plt.scatter(X_train_norm, y_train, c="#1F77B4", edgecolor="#333")
plt.plot(X_test_norm, y_pred, linestyle="--", color='#1F77B4', lw=3)
plt.legend(['Training examples', 'Trained Model'], fontsize=12)
ax.set_xlabel('x', size=12)
ax.set_ylabel('y', size=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid()

# Plot parameter history
ax = fig.add_subplot(1, 2, 2)
plt.plot(ws, lw=3)
plt.plot(bs, lw=3)
plt.legend([r'Weight $w$', r'Bias unit $b$'], fontsize=12)
ax.set_xlabel('Iteration', size=12)
ax.set_ylabel('Value', size=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid()

plt.show()


# ### Keras `fit` function

# In this section, we look at how to obtain a dataset from `tensorflow_datasets` and use it to train a Keras model. In particular we will use the Iris Dataset which consists of 150 observations of the petal and sepal lengths, and petal and sepal widths of 3 different types of irises.
# 

# <!-- ```{figure} ../../img/iris.jpeg
# ---
# name: iris
# ---
# From left to right: [Iris setosa](https://commons.wikimedia.org/w/index.php?curid=170298), [Iris versicolor](https://commons.wikimedia.org/w/index.php?curid=248095), and [Iris virginica](https://www.flickr.com/photos/33397993@N05/3352169862).
# ``` -->

# In[34]:


iris, iris_info = tfds.load('iris', with_info=True)
print(iris_info.splits) 


# In[35]:


type(iris['train'])


# <br>
# 
# **Dataset split.** This only has a train set, so we have to manually split for validation. We can do this with the `.take()` and `.skip()` methods. But this can lead to some unexpected behavior after calling `.shuffle` which converts the dataset to a `ShuffleDataset` which would shuffle the after the initial application of take when creating the train dataset. A workaround is to set `reshuffle_each_iteration` to `False`. 

# In[36]:


tf.random.set_seed(1)

# Shuffle data
dataset_orig = iris['train']
N = len(dataset_orig)
dataset_shuffled = dataset_orig.shuffle(N, reshuffle_each_iteration=False)

# Split into train and test sets; transform
train_dataset = dataset_shuffled.take(100)
test_dataset = dataset_shuffled.skip(100)
print("Train size:", len(train_dataset))
print("Test size: ", len(test_dataset))

train_dataset = train_dataset.map(lambda d: (d['features'], d['label']))
test_dataset = test_dataset.map(lambda d: (d['features'], d['label']))


# <br>
# 
# **Model.** A two-layer MLP with sigmoid activations should suffice to learn 100 data points:

# In[37]:


iris_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='sigmoid', name='fc1', input_shape=(4,)),
    tf.keras.layers.Dense(3, name='fc2', activation='softmax')
])

iris_model.summary() # No need to call .build(), input_shape passed in first dense layer.


# <br>
# 
# **Training.** Observe that the Keras `fit` method works with `tf.data` batch loaders:

# In[38]:


# Use sparse since targets are 0, 1, 2
iris_model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 8

# This will be iterated over in the fit method.
train_loader = train_dataset.shuffle(buffer_size=100)
train_loader = train_loader.batch(batch_size=BATCH_SIZE, drop_remainder=True)
train_loader = train_loader.prefetch(buffer_size=10) # Prepare next elements 
train_loader = train_loader.repeat(NUM_EPOCHS)

# Train model
history = iris_model.fit(
    train_loader, 
    epochs=NUM_EPOCHS,
    steps_per_epoch=np.floor(len(train_dataset) / BATCH_SIZE),
    verbose=1,
)


# In[39]:


hist = history.history

# Train loss plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
ax[0].plot(hist['loss'], lw=2)
ax[0].set_title('Training loss', size=12)
ax[0].set_xlabel('Epoch', size=12)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].grid()

# Accuracy plot
ax[1].plot(hist['accuracy'], lw=2)
ax[1].set_title('Training accuracy', size=12)
ax[1].set_xlabel('Epoch', size=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].grid()

plt.tight_layout()


# <br>
# 
# **Evaluation.** Keras methods `evaluate` (and also `predict`) work nicely with TF dataset objects. We can load the test data into the evaluator as follows:

# In[40]:


results = iris_model.evaluate(test_dataset.batch(1), verbose=0) # 50 batches of size 1
print('Test loss: {:.4f}   Test Acc.: {:.4f}'.format(*results))


# In[ ]:




