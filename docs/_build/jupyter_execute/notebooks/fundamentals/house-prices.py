#!/usr/bin/env python
# coding: utf-8

# # Modelling with Pipelines

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

# Pipelines in the [`scikit-learn` API](https://scikit-learn.org/stable/modules/classes.html#) allow us to apply a sequence of transformers and a final estimator on our dataset. Since each intermediate step implements a `fit` and `transform` method, while the final estimator implements a `fit` method, this setup allows us to recursively fit of all transformations on the dataset in a single step. Another advantage is that the pipeline can be cross-validated as a whole while setting different parameters. 
# 
# From the perspective of code maintainability and testability, pipelines are easier to work with because they allow us to separate declarative from imperative code as shown in {numref}`pipelines`. In the last section of this notebook, we will show that we can easily load pipelines to make inference on test data with little to no preprocessing.
# 
# ```{figure} ../../img/pipelines.png
# ---
# width: 40em
# name: pipelines
# ---
# Pipelines allow us to cleanly separate declarative from imperative code. [[source]](https://gh.mltrainings.ru/presentations/LopuhinJankiewicz_KaggleMercari.pdf)
# ```
# 
# In this notebook, we will look at how to create a complete pipeline in scikit-learn that performs feature engineering, and inference for predicting house prices. We will show how to fine-tune this pipeline as a whole, then perform inference without further modification. 

# ## House prices dataset
# 
# Our task is to use California census data to build a model of housing prices. This data includes metrics such as the population, median income, and median housing price for each district in California. This is a regression problem and will use **root mean squared error** as our evaluation metric.

# ### Downloading the dataset

# Here we download the data using the `request.urlretrieve` function from `urllib`. This function takes a URL where the data is hosted and a save path where the data will be stored on the local disk.

# ```{margin}
# ⚠️ **Attribution:** This notebook follows [Chapter 2](https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb) of {cite}`geron2019hands-on` released under [Apache License 2.0](https://github.com/ageron/handson-ml2/blob/master/LICENSE) for obtaining and preprocessing of the dataset, and feature engineering.
# ```

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tarfile
import urllib.request

from inefficient_networks.config import config
config.set_matplotlib()
config.set_ignore_warnings()


def fetch_housing_data(housing_url, housing_path):
    '''Download data from `housing_url` and save it to `housing_path`.'''
    
    # Make directory
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    
    # Download data in housing_url to tgz_path
    urllib.request.urlretrieve(housing_url, tgz_path)
    
    # Extract tar file
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Downloading the dataset:

# In[2]:


# Dataset URL
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Local save path
HOUSING_PATH = config.DATASET_DIR / "housing"

# Downloading the data
fetch_housing_data(HOUSING_URL, HOUSING_PATH)


# ### Quick look at the data

# Let us load the data using pandas.

# In[3]:


housing = pd.read_csv(HOUSING_PATH / "housing.csv")

print(housing.info())
housing.head()


# The feature `total_bedrooms` is sometimes missing. All features are numerical except `ocean_proximity`.

# In[4]:


housing.ocean_proximity.value_counts()


# The following plot shows how median house value is generally higher near the ocean. Moreover, prices seem to depend on population density.

# In[5]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c=housing["median_house_value"], cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
plt.grid();


# Looking at distribution of values. Distributions are tail-heavy; transforming the data to make it more bell-shaped may help some algorithms. Housing median age and house prices are capped.

# In[6]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# Checking for missing values:

# In[7]:


housing.isna().sum()


# ## Feature transformation pipeline

# In this section we design two pipelines for separate preprocessing of categorical and numerical features of the dataset. These two pipelines will be combined in a feature transformation pipeline for the dataset. For numerical data, we perform simple imputation of missing values with the median value. For the categorical feature `ocean_proximity`, we perform one-hot encoding.

# In[8]:


from sklearn.impute import SimpleImputer

# Fitting the imputer on numerical fetures
housing_num = [f for f in housing.columns if f not in ['ocean_proximity', 'median_house_value']]
imputer = SimpleImputer(strategy='median')


# Note that we set `categories` in the encoder since a fold during cross-validation may contain less than all possible categories, resulting in fewer columns after being transformed. 

# In[9]:


from sklearn.preprocessing import OneHotEncoder

housing_cat = ['ocean_proximity']
categories = [list(housing[cat].unique()) for cat in housing_cat]
cat_encoder = OneHotEncoder(categories=categories, sparse=False)

cat_encoder.fit_transform(housing[housing_cat].iloc[:10, :])


# In[10]:


cat_encoder.categories_ # All categories even if fitted on subset of data


# ### Custom transformers

# Although scikit-learn provides many useful transformers, we will need to write own for tasks such as custom cleanup operations or combining specific
# attributes. Custom transformers work seamlessly with existing scikit-learn functionalities, all we need to do is create a class and implement three methods: `fit()` (that returns `self`) and `transform()` (we get `fit_transform` for free from the `TransformerMixin` as a base class). If we add `BaseEstimator` as a base class, we will also get two extra methods `get_params()` and `set_params()` that will be useful for automatic hyperparameter tuning. Here we create a transformer for adding new features.

# In[11]:


from sklearn.base import BaseEstimator, TransformerMixin

class CombinedFeaturesAdder(BaseEstimator, TransformerMixin):
    """Transformer for adding feature combinations discussed above."""
    
    def __init__(self, add_bedrooms_per_room=False): # No *args, **kwargs (!)
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        rooms_per_household = X["total_rooms"] / X["households"]
        population_per_household = X["population"] / X["households"]
        X["rooms_per_household"] = rooms_per_household
        X["population_per_household"] = population_per_household
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X["total_bedrooms"] / X["total_rooms"]
            X["bedrooms_per_room"] = bedrooms_per_room
        
        return X


# In this example the transformer has one hyperparameter, `add_bedrooms_per_room`,
# set to `False` by default (we don't want the default behavior to change the column count). This hyperparameter will allow us to easily find out whether adding this attribute helps the
# ML algorithms or not.

# In[12]:


feature_adder = CombinedFeaturesAdder(add_bedrooms_per_room=True)
housing_extra_features = feature_adder.transform(housing)


# Checking the shapes, the `feature_adder` should add two columns to the feature set.

# In[13]:


print(housing.shape)
print(housing_extra_features.shape)
housing_extra_features.head()


# ### Arranging the pipeline

# The `Pipeline` constructor takes a list of name estimator pairs defining a sequence of steps. All but the last estimator must be transformers (i.e. they must have a `fit_transform()` method). The names can be anything as long as they are unique and don't contain double underscores. Calling the pipeline's `fit()` method calls `fit_transform()` sequentially on all transformers, passing the output of each call as the parameter to the next call until it reaches the final estimator, for which it calls the `fit()` method.
# 
# The pipeline exposes the same methods as the final estimator. For our preprocessing pipeline, the last estimator is a `StandardScaler` is a transformer, so the pipeline has a `transform` method that applies all the transforms to the data in sequence.

# In[14]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# Feature transformation pipeline
cat_feat = ['ocean_proximity']
num_feat = [f for f in housing.columns if f not in cat_feat + ['median_house_value']]

feature_pipe = ColumnTransformer(
    [
        (
            'num', 
            Pipeline([
                ('feature_adder', CombinedFeaturesAdder()),
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler()),    
            ]), 
            num_feat
        ),
        (
            'cat', 
            Pipeline([
                ('one_hot', OneHotEncoder(categories=categories, sparse=False))
            ]), 
            cat_feat
        )
    ],
    remainder='drop' # Columns not in num_feat + cat_feat are dropped.
)

# Preprocessed dataset
housing_transformed = feature_pipe.fit_transform(housing)
housing_transformed.shape


# ## Optimizing the prediction pipeline

# To make a prediction pipeline, we append the feature transformation pipeline with a regression model.
# Note that we are able to call `fit` on the full pipeline since each step of the feature transformation pipeline implements `fit_transform` method. Thus, training data is sequentially transformed then fed to the model as training data. 
# 
# Observe that this model has hyperparameters that control its design (both hyperparameters for feature engineering, and for the model). Instead of tuning a single model using transformed data, we will tune the entire prediction pipeline using `optuna`. Note that `remainder='drop'` in `ColumnTransformer` means that columns that are not in `num_feat` and `cat_feat` are dropped from the dataset that will be fed into the model.

# In[15]:


import optuna
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


def create_pipe(max_depth, n_estimators, add_bpr):
    '''Initialize full pipeline from given hyperparameters.'''

    # Feature engineering pipeline
    cat_feat = ['ocean_proximity']
    num_feat = [f for f in housing.columns if f not in cat_feat + ['median_house_value']]
    feature_pipe = ColumnTransformer(
        [
            (
                'num', 
                Pipeline([
                    ('feature_adder', CombinedFeaturesAdder(add_bpr)),
                    ('imputer', SimpleImputer(strategy='median')),
                    ('std_scaler', StandardScaler())
                ]), 
                num_feat
            ),
            (
                'cat', 
                Pipeline([
                    ('one_hot', OneHotEncoder(categories=categories, sparse=False))
                ]), 
                cat_feat
            )
        ],
        remainder='drop'
    )

    # Model for transformed dataset
    model = RandomForestRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
    )

    # Assemble prediction pipeline
    return Pipeline(
        [
            ('prep', feature_pipe), 
            ('model', model)
        ]
    )


def objective(trial, X, y):
    '''Average RMSE over cv-folds.'''
    
    pipe = create_pipe(
        add_bpr=trial.suggest_categorical('add_bpr', [True, False]), 
        max_depth=trial.suggest_int('max_depth', 5, 35),
        n_estimators=trial.suggest_int('n_estimators', 200, 1000), 
    )
    
    score = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error')
    return np.sqrt(-score).mean()


# Create study for minimizing the RMSE.
X = housing.drop(['median_house_value'], axis=1)
y = housing.median_house_value.values

study = optuna.create_study(direction='minimize')
study.optimize(partial(objective, X=X, y=y), n_trials=30, n_jobs=-1)


# In[16]:


print("\nBest trial:")
best_trial = study.best_trial

print("  Number:", best_trial.number)
print("  Value: ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))


# Observe that `add_bedrooms_per_room=True` so adding the extra feature is the best choice. In fact this is the choice for most top scoring trials. This shows that even preprocessing steps can be optimized as a hyperparameter of the pipeline.

# In[22]:


study.trials_dataframe().sort_values("value", ascending=True).head(10).reset_index(drop=True)


# In[18]:


fig = optuna.visualization.plot_contour(study, params=["max_depth", "n_estimators"])
fig.update_layout(width=650, height=600)
fig.show(renderer="svg")


# In[19]:


fig = optuna.visualization.plot_slice(study, ["max_depth", "n_estimators"])
fig.update_layout(width=900, height=400)
fig.show(renderer="svg")


# In[28]:


fig = optuna.visualization.plot_parallel_coordinate(study)
fig.update_layout(width=900, height=400)
fig.show(renderer="svg")


# ## Inference using the pipeline

# Assuming the test data is identical to train data, any trained pipeline can be used for inference without any specific preprocessing steps. Note that for the sake of simplicity we did not save the best models during hyperparameter optimization. But this can be done with [callbacks](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html?highlight=callbacks). In that case, we can simply load the best model instead of retraining which we do here. 

# In[23]:


from sklearn.metrics import mean_squared_error

# Fit on best parameters
best_pipe = create_pipe(**study.best_params)
best_pipe.fit(X, y)

# Make inference / prediction
y_pred = best_pipe.predict(X)
print('RMSE (train):', np.sqrt(mean_squared_error(y, y_pred)))


# ## Conclusion
# 
# Pipelines are **awesome**. We can use scikit-learn transformers as well as define custom transformers as steps in the pipeline, making this technique very flexible. For example, even preprocessing can be included in the pipeline, and therefore be part of cross-validation and hyperparameter optimization. Finally, we have shown that the pipeline can be loaded to perform inference directly on new test data with minimal preprocessing. This means that testing, development, and maintenance can be done in one place in the code.

# 
