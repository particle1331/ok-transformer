#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow as tf
import tensorflow.keras as k
from inefficient_networks.config import config

(x_train, y_train), (x_test, y_test) = k.datasets.mnist.load_data()


# In[40]:


config.set_tensorflow_seeds(4)
model = k.Sequential()
model.add(k.layers.Dense(512, 'relu'))
model.add(k.layers.Dense(10))

model.compile(
    metrics='accuracy',
    optimizer='adam',
    loss=k.losses.SparseCategoricalCrossentropy()
)

model.fit(x_train.reshape(-1, 784), y_train, epochs=2, batch_size=32)


# In[41]:


config.set_tensorflow_seeds(4)
model = k.Sequential()
model.add(k.layers.Dense(512, 'relu'))
model.add(k.layers.Dense(10, 'softmax'))

model.compile(
    metrics='accuracy',
    optimizer='adam',
    loss=k.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.fit(x_train.reshape(-1, 784), y_train, epochs=2, batch_size=32)


# In[37]:


tf.reduce_sum(model(x_train.reshape(-1, 784))[3])
# model.fit(x_train.reshape(-1, 784), y_train, epochs=2, batch_size=32)


# In[36]:


k.activations.softmax(model(x_train.reshape(-1, 784)))[3]


# In[26]:


k.activations.sigmoid(model(x_test.reshape(-1, 784)))


# In[ ]:





# In[ ]:




