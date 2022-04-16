#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Estimators

# ```{admonition} Attribution
# This notebook builds on Chapter 14: *Going Deeper – The Mechanics of TensorFlow* of {cite}`RaschkaMirjalili2019`.
# ```

# In this notebook, we will work with TensorFlow Estimators. The `tf.estimator` API encapsulates the underlying steps in machine learning tasks, such as training, prediction (inference), and evaluation. Estimators are more encapsulated but also more scalable when compared to the previous approaches that we have covered above. Also, the `tf.estimator` API adds support for running models on multiple platforms without requiring major code changes, which makes them more suitable for the so-called "production phase" in industry applications. 
# 
# TensorFlow comes with a selection of off-the-shelf estimators for common machine learning and deep learning architectures that are useful for comparison studies, for example, to quickly assess whether a certain approach is applicable to a particular dataset or problem. Besides using pre-made Estimators, we can also create an Estimator by converting a Keras model to an Estimator.

# In[18]:


import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


# ## Working with feature columns

# In machine learning and deep learning applications, we can encounter various
# different types of features: continuous, unordered categorical (nominal), and ordered categorical (ordinal). Note that while numeric data can be either continuous or discrete, in the context of the TensorFlow API, "numeric" data specifically refers to continuous data of the floating point type.
# 
#  Sometimes, feature sets are comprised of a mixture of different feature types. While TensorFlow Estimators were designed to handle all these different types of features, we must specify how each feature should be interpreted by the Estimator.

# ### Auto MPG dataset

# ```{figure} ../../img/feature_cols.png 
# ---
# width: 60em
# name: feature_cols
# ---
# Assigning types to feature columns from the Auto MPG dataset.
# 
# ```

# To demonstrate the use of TF Estimators, we use the [Auto MPG dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg). We are going to treat five features from the Auto MPG dataset (*number of cylinders*, *displacement*, *horsepower*, *weight*, and *acceleration*) as numeric (i.e. continuous) features. The *model year* can be regarded as an ordered categorical feature. Lastly, the *manufacturing origin* can be regarded as an unordered categorical feature with three possible discrete values, 1, 2, and 3, which correspond to the US, Europe, and Japan, respectively. {numref}`feature_cols` above shows how we will treat these feature columns. 

# In[19]:


import pandas as pd
import numpy as np

dataset_path = tf.keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

column_names = [
    "MPG", "Cylinders", "Displacement",
    "Horsepower", "Weight", "Acceleration",
    "ModelYear", "Origin"
]

# Load dataset; drop missing values
df = pd.read_csv(dataset_path, 
    names=column_names, 
    na_values="?", 
    comment="\t", 
    sep=" ", 
    skipinitialspace=True)

print("Shape:", df.shape)
print("No. of missing values:")
print(df.isna().sum())

# For simplicity drop rows with missing values.
df = df.dropna().reset_index(drop=True)
df.tail()


# Splitting the dataset and standardizing numerical columns:

# In[20]:


import sklearn
import sklearn.model_selection

df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8)
train_stats = df_train.describe().transpose()
train_stats


# In[21]:


numeric_column_names = [
    'Cylinders', 
    'Displacement', 
    'Horsepower', 
    'Weight', 
    'Acceleration'
]

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col_name in numeric_column_names:
    train_mean = train_stats.loc[col_name, 'mean']
    train_std  = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - train_mean) / train_std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - train_mean) / train_std
    
df_train_norm.tail()


# ### Numeric features

# In the following code, we will use TensorFlow's `feature_column` function
# to transform the 5 continuous features into the feature column data structure that
# TensorFlow Estimators can work with:

# In[29]:


numeric_features = []
for col_name in numeric_column_names:
    feature_column = tf.feature_column.numeric_column(key=col_name)
    numeric_features.append(feature_column)

print(numeric_features)


# ### Bucketized features

# Next, let's group the rather fine-grained model year information into buckets to
# simplify the learning task for the model that we are going to train later. Note that we assign `boundaries=[73, 76, 79]` which results in left-closed partitioning of the real line into 4 intervals `(-∞, 73)`, `[73, 76)`, `[76, 79)`, and `[79, +∞)`.

# In[28]:


feature_year = tf.feature_column.numeric_column(key="ModelYear")
bucketized_column = tf.feature_column.bucketized_column(
    source_column=feature_year,
    boundaries=[73, 76, 79]
)

# For consistency, we create list of bucketized features
bucketized_features = [] 
bucketized_features.append(bucketized_column)
print(bucketized_features)


# ### Categorical indicator features

# Next, we will proceed with defining a list for the unordered categorical feature,
# `Origin`. Here we use `categorical_column_with_vocabulary_list` in `tf.feature_column` and provide a list of all possible category names as input. 
# 
# ```{tip}
# If the list of possible categories is too large, we can use `categorical_column_with_vocabulary_list` and provide a file that contains all the categories/words so that we do not have to store a list of all possible words in memory.  If the features are already associated
# with an index of categories in the range `[0, num_categories)`, then we can use the
# `categorical_column_with_identity` function. However,
# in this case, the feature `Origin` is given as integer values `1`, `2`, `3` (as opposed to `0`, `1`, `2`), which does not match the requirement for categorical indexing.
# ```

# In[25]:


print(df.Origin.unique())


# In[30]:


feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(
    key='Origin',
    vocabulary_list=[1, 2, 3]
)


# ```{margin}
# Refer to the [official TensorFlow docs](https://www.tensorflow.org/api_docs/python/tf/feature_column) for other implemented feature columns such as hashed columns and crossed columns.
# ```
# 
# Certain Estimators, such as `DNNClassifier` and `DNNRegressor`, only accept so-called
# "dense columns." Therefore, the next step is to convert the existing categorical feature column to such a dense column. There are two ways to do this: using an embedding column via `embedding_column` or an indicator column via `indicator_column`. We use the latter which converts the categorical indices to one-hot encoded vectors to convert the categorical column to a dense format:

# In[32]:


indicator_column = tf.feature_column.indicator_column(feature_origin)

# For consistency, we create list of nominal features
categorical_indicator_features = []
categorical_indicator_features.append(indicator_column)
print(categorical_indicator_features)


# ## Machine learning with pre-made estimators

# ### Input functions
# 
# We have to define an **input function** that 
# processes the data and returns a TensorFlow dataset consisting of a tuple 
# that contains the input features and the targets. Note that the features 
# must be a dictionary format such that the keys match 
# the names (or keys) of feature columns.

# In[42]:


def train_input_fn(df_train, batch_size=8):
    df = df_train.copy()
    x_train, y_train = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train))

    # Shuffle, batch, and repeat the examples
    return dataset.shuffle(1000).batch(batch_size).repeat()

# Inspection
ds = train_input_fn(df_train_norm)
batch = next(iter(ds))
print('Keys:', batch[0].keys())
print('Batch Model Years:', batch[0]['ModelYear'])
print('Batch MPGs (targets):', batch[1].numpy())


# Input function for evaluation:

# In[43]:


def eval_input_fn(df_eval, batch_size=8):
    df = df_eval.copy()
    x_eval, y_eval = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(x_eval), y_eval))

    # Shuffle, batch, and repeat the examples
    return dataset.shuffle(1000).batch(batch_size).repeat()

# Inspection
ds = eval_input_fn(df_test_norm)
batch = next(iter(ds))
print('Keys:', batch[0].keys())
print('Batch Model Years:', batch[0]['ModelYear'])
print('Batch MPGs (targets):', batch[1].numpy())


# ### Initializing the Estimator

# Since predicting MPG values
# is a typical regression problem, we will use `tf.estimator.DNNRegressor`. When
# instantiating the regression Estimator, we will provide the list of feature columns
# and specify the number of hidden units that we want to have in each hidden layer
# using the argument `hidden_units`.

# In[47]:


regressor = tf.estimator.DNNRegressor(
    feature_columns=(
        numeric_features + 
        bucketized_features + 
        categorical_indicator_features
    ),
    hidden_units=[32, 10],
    model_dir='models/autompg-dnnregressor/')


# The other argument, `model_dir`, that we have provided specifies the directory
# for saving model parameters. One of the advantages of Estimators is that they
# automatically checkpoint the model during training, so that in case the training of
# the model crashes for an unexpected reason, we can easily load
# the last saved checkpoint and continue training from there. The checkpoints will also
# be saved in the directory specified by `model_dir`.

# ### Training
# 
# The `.train()` method expects two arguments. The argument `input_fn` expects a callable that returns a batch of training examples. The `steps` which is the total number of SGD updates (or calls to the input function) is calculated as follows:

# In[50]:


EPOCHS = 30
BATCH_SIZE = 8
total_steps = EPOCHS * int(np.ceil(len(df_train) / BATCH_SIZE))
print('Training Steps:', total_steps)

regressor.train(
    input_fn=lambda: train_input_fn(df_train_norm, batch_size=BATCH_SIZE),
    steps=total_steps
)


# ```{note}
# 
# Recall that `model_dir` saves the checkpoints of the model during training. The last model can be loaded using the `warm_start_from` argument as follows:
# 
# ```python
# reloaded_regressor = tf.estimator.DNNRegressor(
#     feature_columns=all_feature_columns,
#     hidden_units=[32, 10],
#     warm_start_from='models/autompg-dnnregressor/',
#     model_dir='models/autompg-dnnregressor/'
# )
# ```
# 

# ### Evaluation

# To evaluate performance, we use the `.evaluate` method:

# In[54]:


eval_results = regressor.evaluate(
    input_fn=lambda: eval_input_fn(df_test_norm, batch_size=8)
)


# In[ ]:


print(eval_results)


# In[17]:


pred_res = regressor.predict(input_fn=lambda: eval_input_fn(df_test_norm, batch_size=8))

print(next(iter(pred_res)))


# In[18]:


tf.get_logger().setLevel('ERROR')


# In[19]:


boosted_tree = tf.estimator.BoostedTreesRegressor(
    feature_columns=all_feature_columns,
    n_batches_per_layer=20,
    n_trees=200)

boosted_tree.train(
    input_fn=lambda:train_input_fn(df_train_norm, batch_size=BATCH_SIZE))

eval_results = boosted_tree.evaluate(
    input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8))

print(eval_results)

print('Average-Loss {:.4f}'.format(eval_results['average_loss']))


# In[ ]:




