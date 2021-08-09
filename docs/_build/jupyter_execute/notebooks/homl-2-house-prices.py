#!/usr/bin/env python
# coding: utf-8

# # End-to-End ML Project

# ```{admonition} Attribution
# This notebook is based on Chapter 2 of {cite}`geron2019hands-on`.
# ```

# ## Looking at the big picture
# 
# In this notebook, we will work through an example project end to end, pretending to be a recently hired data scientist at a real estate company. Our first task is to use California census data to build a model of housing prices in the state for a real estate investing company. This data includes
# metrics such as the population, median income, and median housing price for each
# block group in California. Block groups are the smallest geographical unit for which
# the US Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people). We will call them “districts” for short.
# Your model should learn from this data and be able to predict the **median housing price** in any district, given all the other metrics.
# 
# How does the company expect to use and benefit
# from this model? Building a
# model is probably not the end goal. Knowing the objective is important because it will determine how
# you frame the problem, which algorithms you will select, which performance measure you will use to evaluate your model, and how much effort you will spend tweaking it. It turns out that your model’s output (a prediction of a district’s median housing price) will be fed to another Machine Learning system, along
# with many other signals ({numref}`ml_pipeline`). This downstream system will determine whether it is worth investing in a given area or not. Getting this right is critical, as it directly affects revenue. 
# 
# It is therefore imporant to ask what the downstream system does to the output of our model. If the downstream system converts the prices into categories (e.g., “cheap,” “medium,” or “expensive”) and then uses those categories instead of the prices themselves, then getting the price perfectly right is not important at all; your system just needs to get the category right. If that’s so, then the problem should have been framed as a classification task, not a regression task. You don’t want to find
# this out after working on a regression system for months.
# 
# Finally, we need to be able to assess our model's success, this can come in the form of a **baseline** performance. The current district housing prices are currently estimated manually by experts: a team gathers up-to-date information about a district, and when they cannot get the median housing price, they
# estimate it using complex rules. This is costly and time-consuming, and their estimates are not great; in cases where
# they manage to find out the actual median housing price, they often realize that their
# estimates were off by **more than 20%**. So our model must perform at a better error rate.
# 

# ```{figure} ../img/ml_pipeline.png
# ---
# width: 45em
# name: ml_pipeline
# ---
# A Machine Learning pipeline for real estate investments. Part of the pipeline is the District Pricing model that is currently being developed.
# ```
# 

# ```{admonition} Pipelines
# ---
# class: note
# ---
# 
# A sequence of data processing components is called a **data pipeline**. Pipelines are very
# common in Machine Learning systems, since there is a lot of data to manipulate and
# many data transformations to apply.
# 
# * Components typically run asynchronously. Each component pulls in a large amount
# of data, processes it, and spits out the result in another data store. Then, some time
# later, the next component in the pipeline pulls this data and spits out its own output.
# +++
# * Each component is fairly self-contained: the interface between components is simply
# the data store. This makes the system simple to grasp (with the help of a data flow
# graph), and different teams can focus on different components.
# +++
# * Moreover, if a component breaks down, the downstream components can often continue to run normally (at least for a while) by just using the last output from the broken component.
# This makes the architecture quite robust. On the other hand, a broken component can go unnoticed for some time if proper
# monitoring is not implemented. The data gets stale and the overall system’s perfor‐
# mance drops.
# 
# ```

# ## Getting the data

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

import os
from pathlib import Path
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
DATA_DIR = Path().resolve().parents[1] / 'data'
HOUSING_PATH = DATA_DIR / "housing"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
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


# Download the data.

# In[2]:


fetch_housing_data()


# ### Quick look

# Let's load the data using pandas.

# In[3]:


housing = pd.read_csv(HOUSING_PATH / "housing.csv")
housing.head()


# In[4]:


housing.shape


# Each row represents one district which is described by 10 features (including the target).

# In[5]:


housing.info()


# The feature `total_bedrooms` is sometimes missing. All features numerical except `ocean_proximity` which is text.

# In[6]:


housing.ocean_proximity.value_counts()


# In[7]:


housing.describe() # statistics of numerical features


# We can better visualize this table using histograms.

# In[8]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# Some observations and metadata.
# 
# * Median income are clipped between 0.5 and 15. This is measured in roughly $10,000 units. 
# 
# +++
# 
# * Housing median age and median house values are also capped. The latter may be a serious problem since it is our target. Model may learn that prices never go beyond the maximum value. Two options: (1) gather more data for districts whose labels are capped, or (2) remove these from training set (and also from the test set, since the model should not be evaluated poorly for districts outside this range).
# 
# +++
# 
# * These attributes have very different scales. Depending on the algorithm, we might need to perform feature scaling.
# 
# +++
# 
# * Many histograms are tail-heavy: they extend much farther to the right of the median than to the left. This may make it a bit harder for some ML algorithms to detect patterns. We will try transforming these attributes later on to have more bell-shaped distributions.

# ### Creating a test set

# We can consider purely random sampling methods using `train_test_split` from `sklearn.model_selection`. This is generally fine if the dataset is large enough (especially relative to the number of attributes), but if it
# is not, we run the risk of introducing a significant sampling bias.
# 
# Suppose we chatted with experts who told us that the median income is a very
# important attribute to predict median housing prices. We may then perform *stratified sampling* based on income categories to ensure that the test set is representative of the various categories of incomes in the whole dataset. 

# In[9]:


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist();


# In[10]:


housing["income_cat"].value_counts(normalize=True)


# In[11]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# We check the percentage error based on the `income_cat` distribution of the whole dataset.

# In[12]:


uniform_error = (test_set['income_cat'].value_counts(normalize=True) - housing["income_cat"].value_counts(normalize=True))
uniform_error / housing["income_cat"].value_counts(normalize=True)


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set  = housing.loc[test_index]
    

strat_error = strat_test_set['income_cat'].value_counts(normalize=True) - housing["income_cat"].value_counts(normalize=True)
strat_error / housing["income_cat"].value_counts(normalize=True)


# This looks way better. We drop the temporary feature `income_cat` from the train and test sets.

# In[14]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# ## Data visualization

# In[15]:


housing = strat_train_set.copy()


# ### Geographical data

# In[16]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend();


# **Fig.** California housing prices: red is expensive, blue is cheap, larger circles indicate areas with a larger population

# This image tells you that the housing prices are very much related to the location
# (e.g., close to the ocean) and to the population density, as you probably knew already.
# A clustering algorithm should be useful for detecting the main cluster and for adding
# new features that measure the proximity to the cluster centers. The ocean proximity
# attribute may be useful as well, although in Northern California the housing prices in
# coastal districts are not too high, so it is not a simple rule.

# ### Looking for correlations

# Let us look at correlations of ther target with other features.

# In[17]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False) 


# Looks like our experts are correct in saying that median income is predictive of median house value.

# :::{caution}
# The correlation coefficient only measures linear correlations. It may completely miss
# out on meaningful nonlinear relationships.
# :::

# In[18]:


from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8));


# This matrix is symmetric. So we can focus our attention, on the upper half. In particular, we look at the plot `median_house_value` vs `median_income`. This reveals a few things:
# * There is a clear upward trend and the values are not too dispersed. 
# 
# +++
# 
# * The plot clearly shows that the house value is clipped at around 500,000. But notice that there are horizontal lines around 350,000 and 450,000. Perhaps these are round numbers that occur across the median income range, and are determined by other features. We may want to try removing the corresponding districts to prevent your algorithms from learning to reproduce these data quirks.

# ### Feature combinations

# We experiment with feature combinations. Here, we look at ratios. These are generally nice, since these are naturally scale free attributes.

# In[19]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]   
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"] 
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# The new `bedrooms_per_room` attribute is much more correlated with
# the median house value than the total number of rooms or bedrooms. Apparently
# houses with a lower bedroom/room ratio tend to be more expensive. This makes sense. The number of
# rooms per household `rooms_per_household` is also more informative than the total number of rooms `total_rooms` in a
# district—obviously the larger the houses, the more expensive they are.

# ## Preprocessing

# Revert back to the original stratified train set:

# In[20]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# ### Data cleaning

# Three options for dealing with missing features:
# 1. Get rid of the corresponding districts (drop rows).
# 2. Get rid of the whole feature (drop column).
# 3. Imputation with some derived value (zero, mean, median, etc.).
# 
# ```python
# housing.dropna(subset=["total_bedrooms"])     # option 1
# housing.drop("total_bedrooms", axis=1)        # option 2
# median = housing["total_bedrooms"].median()   # option 3
# housing["total_bedrooms"].fillna(median, inplace=True)
# ```

# We separate processing of numerical and categorical variables, then perform median imputation on numerical features.

# In[21]:


from sklearn.impute import SimpleImputer

housing_num = housing[[f for f in housing.columns if f != 'ocean_proximity']]
housing_cat = housing[['ocean_proximity']]

# Fitting the imputer on numerical fetures
imputer = SimpleImputer(strategy='median')
imputer.fit(housing_num)

# Checking...
(imputer.statistics_ == housing_num.median().values).all()


# Finally, we check that there are no more null values in the datasets

# In[22]:


X = imputer.transform(housing_num)
housing_num_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

housing_num_tr.info()


# In[23]:


housing_cat.info()


# ### Categorical features

# We one-hot encode the `ocean_proximity` feature.

# ```{margin} 
# Learning embeddings is
# an example of **representation learning**. (See Chapters 13 and 17 for
# more details.)
# ```
# 
# :::{note}
# If a categorical attribute has a large number of possible categories
# (e.g. country code, profession, species), then one-hot encoding will
# result in a large number of input features. This may slow down
# training and degrade performance by increasing the dimensionality of each training instance. 
# 
# If this happens, you may want
# to replace the categorical input with useful numerical features
# related to the categories: for example, you could replace the
# `ocean_proximity` feature with the distance to the ocean (similarly,
# a country code could be replaced with the country’s population and
# GDP per capita). Alternatively, you could replace each category
# with a learnable, low-dimensional vector called an **embedding**. Each
# category’s representation would be learned during training. 
# :::

# In[24]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot # sparse


# In[25]:


housing_cat_1hot.toarray() # dense


# In[26]:


cat_encoder.categories_ # learned categories


# ### Custom transformers

# Although Scikit-Learn provides many useful transformers, you will need to write
# your own for tasks such as custom cleanup operations or combining specific
# attributes. You will want your transformer to work seamlessly with Scikit-Learn functionalities (such as pipelines), all you need to do is create a class and implement three methods: `fit()`
# (returning `self`), `transform()`, and `fit_transform()`.
# 
# You can get the last one for free by simply adding `TransformerMixin` as a base class.
# If you add `BaseEstimator` as a base class (and avoid `*args` and `**kwargs` in your constructor), you will also get two extra methods (`get_params()` and `set_params()`) that
# will be useful for automatic hyperparameter tuning.
# 

# In[27]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6 # column indices

class CombinedFeaturesAdder(BaseEstimator, TransformerMixin):
    """Transformer for adding feature combinations discussed above."""
    
    def __init__(self, add_bedrooms_per_room=True): # no *args, **kwargs ! (?)
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        


# In this example the transformer has one hyperparameter, add_bedrooms_per_room,
# set to True by default (it is often helpful to provide sensible defaults). This hyperparameter will allow you to easily find out whether adding this attribute helps the
# ML algorithms or not. For now, we set this to `False`.

# In[28]:


feature_adder = CombinedFeaturesAdder(add_bedrooms_per_room=False)
housing_extra_features = feature_adder.transform(housing.values)


# Checking the shapes, the `feature_adder` should add two columns to the feature set.

# In[29]:


housing.shape


# In[30]:


housing_extra_features.shape


# ### Feature scaling

# With few exceptions, ML algorithms don’t perform well when
# the input numerical attributes have very different scales. This is the case for the housing data: the total number of rooms ranges from about 6 to 39,320, while the median incomes only range from 0 to 15. Note that scaling the target values is generally not required.

# There are two common ways to get all attributes to have the same scale: 
# * min-max scaling 
# * standardization
# 
# Min-max scaling results in values in the range $[0, 1]$. Scikit-Learn provides a transformer called `MinMaxScaler` for this. It has a `feature_range` hyperparameter that lets
# you change the range if, for some reason, you don't want 0–1. Note that this scaling method is sensitive to outliers. 
# 
# Standardization is different: first it subtracts the mean value (so standardized values
# always have a zero mean), and then it divides by the standard deviation so that the
# resulting distribution has unit variance. Unlike min-max scaling, standardization
# does not bound values to a specific range, which may be a problem for some algorithms. However, standardization is much less affected by outliers. Scikit-Learn provides a transformer called `StandardScaler` for
# standardization.

# ### Transformation pipelines

# We use the sklearn `Pipeline` object to combine all preprocessing steps.
# 
# * The `Pipeline` constructor takes a list of name/estimator pairs defining a sequence of
# steps. The estimators need not be instantiated to a Python variable. They can be referenced using the name string, e.g. `imputer` for the `SimpleImputer` object. 
# 
# +++
# 
# * All but the last estimator must be transformers (i.e., they must have a
# `fit_transform()` method). The names can be anything as long as they are
# unique and don’t contain double underscores, `__`; they will come in handy later for
# hyperparameter tuning.
# 
# +++
# 
# * When you call the pipeline’s `fit()` method, it calls `fit_transform()` sequentially on
# all transformers, passing the output of each call as the parameter to the next call until
# it reaches the final estimator, for which it calls the `fit()` method.
# 
# +++
# 
# * The pipeline exposes the same methods as the final estimator. In this example, the last
# estimator is a `StandardScaler`, which is a transformer, so the pipeline has a `transform()` method that applies all the transforms to the data in sequence (and of course also a `fit_transform()` method, which is the one we used).
# 
# We can separate the processing of subsets of columns of the data, then use `ColumnTransformer` to combine everything in a single pipeline. This is done below in defining `full_pipeline`. 

# In[31]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_adder', CombinedFeaturesAdder()), # add_bedrooms_per_room=True
    ('std_scaler', StandardScaler()),    
])

cat_pipeline = Pipeline([('one_hot', OneHotEncoder())])


# Combine to a full pipeline
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape


# :::{note}
# `OneHotEncoder` returns a sparse matrix, while the `num_pipeline` returns
# a dense matrix. When there is such a mix of sparse and dense matrices, the `ColumnTransformer` estimates the density of the final matrix (i.e., the ratio of nonzero
# cells), and it returns a sparse matrix if the density is lower than a given threshold (by
# default, `sparse_threshold=0.3`). In this example, it returns a dense matrix.
# :::

# In[32]:


housing_prepared[:1]


# In[33]:


full_pipeline.sparse_threshold


# :::{caution}  
# `ColumnTransformer` has an argument with default `remainder='drop'` which means non-specified columns in the list of transformers are dropped. Instead, we can specify `remainder='passthrough'` where all remaining columns are skipped and  concatenated with the output of the transformers.
# :::

# ## Model selection

# We train and evaluate three models: **Linear Regression**, **Decision Forest**, and **Random Forest**. Since we don't have a separate validation set, we will evaluate the models using **cross-validation**. 
# 
# In general, we try out
# many models from various categories of ML algorithms (e.g.
# SVMs with different kernels, and possibly a neural network), 
# without spending too much time tweaking the hyperparameters. 
# The goal is to shortlist a few (two to five) promising models.

# ```{margin}
# For **scoring** functions, e.g. accuracy, higher is better. For **cost** or **loss** functions, e.g. RMSE, lower is better. 
# In this case, we use the negative RMSE to flip the graph and convert it to a scoring function.
# ```

# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error 
from sklearn.model_selection import cross_val_score


for model in (LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(random_state=42)):

    # Fit with default parameters
    model.fit(housing_prepared, housing_labels)

    # Checking performance on train set
    predict_labels = model.predict(housing_prepared)
    print(model)
    print('Train RMSE:', np.sqrt(mean_squared_error(housing_labels, predict_labels)))
    print('Train MAPE:', mean_absolute_percentage_error(housing_labels, predict_labels)) # (y_true, y_pred)
    
    # Cross-validation over 10 folds
    scores = cross_val_score(model, housing_prepared, housing_labels,
                             scoring="neg_mean_absolute_percentage_error", cv=10)

    cv_scores = np.sqrt(-scores) # -scores = RMSE
    print(cv_scores.mean(), cv_scores.std())
    print()


# ```{list-table} Train and 10-fold cross-validation MAPE of the models.
# :header-rows: 1
# 
# * - 
#   - LinearRegression
#   - DecisionTreeRegressor
#   - RandomForestRegressor
# * - **Train**
#   - 0.285
#   - 0.000
#   - 0.066
# * - **Valid** 
#   - 0.534 $\pm$ 0.007
#   - 0.495 $\pm$ 0.006
#   - 0.425 $\pm$ 0.008
# ```
# 

# Random Forest looks very promising. However, note that
# the score on the training set is still much lower than on the validation sets, meaning
# that the model is still **overfitting** the training set. Possible solutions for overfitting are
# to simplify the model, constrain or **regularize** it, or get a lot more training data.

# :::{tip}
# You should save every model you experiment with so that you can
# come back easily to any model you want. Make sure you save both
# the hyperparameters and the trained parameters, as well as the
# cross-validation scores and perhaps the actual predictions as well.
# 
# This will allow you to easily compare scores across model types,
# and compare the types of errors they make. You can easily save
# Scikit-Learn models by using Python’s `pickle` module or by using
# the `joblib` library, which is more efficient at serializing large
# NumPy arrays:
# 
# ```python
# import joblib
# 
# joblib.dump(my_model, "my_model.pkl")
# 
# # and later...
# my_model_loaded = joblib.load("my_model.pkl")
# ```
# :::

# ## Fine tuning a model

# One option would be to fiddle with the hyperparameters manually, until you find a
# great combination of hyperparameter values. This would be very tedious work, and
# you may not have time to explore many combinations. But it also builds intuition on what ranges work. 
# 
# If we have a good idea of the search space, then we can start performing **random search** which evaluates a (sufficiently large) fixed number of points on a hyperparameter lattice sampled randomly (uniformly, or according to some distribution). Finally, to squeeze every last drop of performance, we can do **grid search** by evaluating *each* point on a small region of the lattice. 

# ```{figure} ../img/grid-random-search.png
# ---
# width: 45em
# name: grid-random
# ---
# Comparing grid and random search over for finding a good set of hyperparameters.
# ```
# 

# ### Random search

# Suppose we have familiarized ourselves with the hyperparameter search space giving us an idea on the regions to sample. The following `param_grid` tells the program to first evaluate 10 randomly sampled points out of the **union** of all 14 × 6 = 84 combinations of
# `n_estimators` and `max_features` hyperparameter values specified in the first dict, plus all 12 × 9 = 96 combinations of hyperparameter values in the
# points in the second dict, this time with the `bootstrap` hyperparameter set to `False` instead of
# `True` (default).

# In[35]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

start = time.time()
param_grid = [
    {'n_estimators': range(3, 30, 2), 'max_features': range(2, 8)},
    {'n_estimators': range(8, 32, 2), 'max_features': range(2, 20, 2), 'bootstrap': [False], },
]

forest_reg = RandomForestRegressor()
random_search = RandomizedSearchCV(forest_reg, param_grid, cv=2, n_iter=10, random_state=42,
                                   scoring='neg_mean_squared_error',  # we use neg-RMSE more stable
                                   return_train_score=True)

random_search.fit(housing_prepared, housing_labels);
print(time.time() - start)


# In[36]:


pd.DataFrame(random_search.cv_results_).sort_values('rank_test_score')


# In[37]:


random_search.best_params_ # rank 1 in above table


# In[38]:


random_search.best_estimator_


# ### Grid Search

# From them the results of random search, it looks like `param_bootstrap` should be set to `False`. Moreover, the `n_estimators` is between 15 and 30, while `max_features` is between 6 to 10. We perform a finer search by doing grid search on these ranges.

# In[39]:


pd.DataFrame(random_search.cv_results_).sort_values('rank_test_score')[
    ['param_n_estimators', 'param_max_features', 'param_bootstrap', 'rank_test_score', 'mean_test_score']][:4].reset_index()


# In[40]:


start = time.time()
param_grid = [
    {'n_estimators': range(15, 33, 2), 'max_features': range(6, 11, 2), 'bootstrap': [False]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=2,
                           scoring='neg_mean_squared_error',  # we use neg-RMSE more stable
                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels);
print(time.time() - start)


# Let us see whether we found better hyperparameters with this finer search method.

# In[41]:


pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score')[
    ['param_n_estimators', 'param_max_features', 'param_bootstrap', 'rank_test_score', 'mean_test_score']][:4].reset_index()


# The top four are actually better scores that the best score for random search! 

# In[42]:


grid_search.best_estimator_


# :::{tip}
# 
# Turns out, we can fit whole prediction pipelines in `GridSearchCV` and `RandomSearchCV` instead of just a single model. This requires keys for dictionaries in `param_grid` which is better shown than described (basically a recursive application of `__` which explains why we can only use single underscores for names of pipeline elements):
# 
# ```python
# # Append estimator to preprocessing pipeline
# prediction_pipeline = Pipeline([
#     ("preprocessing", full_pipeline),
#     ("rf", best_model)
# ])
# 
# # Specify parameter lattice
# param_grid = [{
#     'preprocessing__num__feature_adder__add_bedrooms_per_room': [True, False], 
#     'rf__n_estimators': [6, 12, 30], 
#     'rf__bootstrap': [True, False],
#     }]
# 
# # Hyperparameter search
# grid_search = GridSearchCV(prediction_pipeline, param_grid, cv=2,
#                            scoring='neg_mean_squared_error',  # we use neg-RMSE more stable
#                            return_train_score=True)
# 
# grid_search.fit(housing, housing_labels)
# grid_search.best_estimator_
# ```
# 
# This returns the following estimator. Observe that `add_bedrooms_per_room=False` is the better parameter setting. This example shows that even preprocessing steps can be optimized as part of sklearn's pipeline architecture. Neat!
# 
# ```python
# Pipeline(steps=[('preprocessing',
#                  ColumnTransformer(transformers=[('num',
#                                                   Pipeline(steps=[('imputer',
#                                                                    SimpleImputer(strategy='median')),
#                                                                   ('feature_adder',
#                                                                    CombinedFeaturesAdder(add_bedrooms_per_room=False)),
#                                                                   ('std_scaler',
#                                                                    StandardScaler())]),
#                                                   ['longitude', 'latitude',
#                                                    'housing_median_age',
#                                                    'total_rooms',
#                                                    'total_bedrooms',
#                                                    'population', 'households',
#                                                    'median_income']),
#                                                  ('cat',
#                                                   Pipeline(steps=[('one_hot',
#                                                                    OneHotEncoder())]),
#                                                   ['ocean_proximity'])])),
#                 ('rf',
#                  RandomForestRegressor(bootstrap=False, max_features=6,
#                                        n_estimators=30))])
# ```
# :::

# ### Feature importance

# You will often gain good insights on the problem by inspecting the best models. For
# example, the RandomForestRegressor can indicate the relative importance of each
# attribute for making accurate predictions:

# In[43]:


feature_importances = grid_search.best_estimator_.feature_importances_

num_attribs = list(housing_num)
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_attribs = list(full_pipeline.named_transformers_["cat"]['one_hot'].categories_[0])
attributes = num_attribs + extra_attribs + cat_one_attribs

feature_importance_df = pd.DataFrame(feature_importances, index=attributes, columns=['importance'])
feature_importance_df.sort_values('importance').plot.bar(figsize=(12, 6));


# ### Evaluating on the test set

# Before evaluating on the test set, we should also look at the specific errors that the system makes, then try to understand 
# why it makes them and what could fix the problem (adding extra features or getting rid of 
# uninformative ones, cleaning up outliers, etc.). Instead we proceed directly to evaluation on the test set, and perform the error analysis there. (Not best practice, of course. But we do it only as an exercise.)

# In[44]:


best_model = grid_search.best_estimator_

targets = strat_test_set['median_house_value']
preds = best_model.predict(full_pipeline.transform(strat_test_set.drop('median_house_value', axis=1)))
print('RMSE:', np.sqrt(mean_squared_error(targets, preds)))
print('MAPE:', mean_absolute_percentage_error(targets, preds))


# This is better than the human experts baseline! 

# In[45]:


plt.figure(figsize=(10, 10))
plt.scatter(targets, preds, alpha=0.4);
plt.plot(range(0, 500000, 10000), range(0, 500000, 10000), 'k--')
plt.xlabel('targets')
plt.ylabel('predictions');


# The model performs badly for targets at 500,000. The model predicts below this value, which is understandable from the data. From the plot, we can observe the model performs best when predicting up to 300,000. For predictions beyond 300,000 actual values are more dispersed. This might have to do with having little data in this region as the following plot shows. (See also tip in the monitoring section about having dedicated subsets for error analysis.)

# In[46]:


strat_train_set['median_house_value'].hist(bins=100);


# Finally, we calculate the [95% confidence interval](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Confidence_Intervals/BS704_Confidence_Intervals_print.html) for the generalization error using `scipy.stats.t.interval`.

# In[47]:


from scipy import stats

confidence = 0.95
squared_errors = (preds - targets) ** 2
mean = squared_errors.mean()
m = len(squared_errors)

np.sqrt(stats.t.interval(confidence, m - 1,
                         loc=np.mean(squared_errors),
                         scale=stats.sem(squared_errors)))


# ## Launch, monitor, and maintain

# Now comes the project prelaunch phase: you need to present your solution (highlighting what you have learned, what worked and what did not, what assumptions were made, and what your system's limitations are), document everything, and create nice presentations with clear visualizations and easy-to-remember statements (e.g., "the median income is the number one predictor of housing prices"). In our case, we found that the best RF model has an MAPE of 17.4% &mdash; significantly better than expert's price estimates. Launching the model means that we can confidently free up some time for the experts so they can work on more interesting and productive tasks. 
# 
# Perfect, you got approval to launch! You now need to get your solution ready for production (e.g., polish the code, write documentation and tests, and so on). Then you can deploy your model to your production environment. One way to do this is to save the trained Scikit-Learn model (e.g., using `joblib`), including the full preprocessing
# and prediction pipeline, then load this trained model within your production environment and use it to make predictions by calling its `predict()` method.
# 
# ### Model serving
# 
# Models can be served within a dedicated web service that downstream applications can
# query through a REST API. Or the models can be run in a dedicated server which stores its prediction in a downstream data store (e.g. a database). Another popular strategy is to deploy your model on the cloud, for example on Google Cloud AI Platform (formerly known as Google Cloud ML Engine): just save your
# model using joblib and upload it to Google Cloud Storage (GCS), then head over to
# Google Cloud AI Platform and create a new model version, pointing it to the GCS
# file. That’s it! This gives you a simple web service that takes care of load balancing and
# scaling for you. It take JSON requests containing the input data (e.g., of a district) and
# returns JSON responses containing the predictions. You can then use this web service
# in your website (or whatever production environment you are using). 
# 
# ### Monitoring
# 
# But deployment is not the end of the story. You also need to write monitoring code to
# check your system's live performance at regular intervals and trigger alerts when it
# drops. This could be a steep drop, likely due to a broken component in your infrastructure, but be aware that it could also be a gentle decay that could easily go unnoticed for a long time. This is quite common because models tend to “rot” over time:
# indeed, the world changes, so if the model was trained with last year’s data, it may not
# be adapted to today's data.
# 
# In some cases, the model’s performance can be inferred from downstream
# metrics. For example, if your model is part of a recommender system and it suggests
# products that the users may be interested in, then it’s easy to monitor the number of
# recommended products sold each day. If this number drops (compared to nonrecommended products), then the prime suspect is the model. This may be because
# the data pipeline is broken, or perhaps the model needs to be retrained on fresh data. 
# 
# However, it's not always possible to determine the model’s performance without any
# human analysis. This depends heavily on the nature of the task. How can you get an
# alert if the model's performance drops, before thousands of defective products get
# shipped to your clients? One solution is to send to human raters a sample of all the
# pictures that the model classified (especially pictures that the model has low confidence). Depending on the task, the raters may need to be experts, or they could be
# nonspecialists, such as workers on a crowdsourcing platform. In some applications they could even be the users themselves, responding for example via surveys or repurposed captchas.[^footnote1]
# 
# Either way, you need to put in place a monitoring system (with or without human
# raters to evaluate the live model), as well as all the relevant processes to define what to
# do in case of failures and how to prepare for them. Unfortunately, this can be a lot of
# work. In fact, it is often much more work than building and training a model.
# 
# If the data keeps evolving, you will need to update your datasets and retrain your
# model regularly. You should probably automate the whole process as much as possible. Here are a few things you can automate:
# 
# * Collect fresh data regularly and label it (e.g., using human raters).
# 
# +++
# 
# * Write a script to train the model and fine-tune the hyperparameters automatically. This script could run automatically, for example every day or every week, depending on your needs.
# 
# +++
# 
# * Write another script that will evaluate both the new model and the previous model on the updated test set, and deploy the model to production if the performance has not decreased (if it did, make sure you investigate why).
# 
# You should also make sure you evaluate the model’s input data quality. Sometimes
# performance will degrade slightly because of a poor-quality signal (e.g., a malfunctioning sensor sending random values, or another team’s output becoming stale), but
# it may take a while before your system’s performance degrades enough to trigger an
# alert. If you monitor your model’s inputs, you may catch this earlier. For example, you
# could trigger an alert if more and more inputs are missing a feature, or if its mean or
# standard deviation drifts too far from the training set, or a categorical feature starts
# containing new categories. (!)
# 
# :::{tip}
# You may want to create several subsets of the test set in order to
# evaluate how well your model performs on specific parts of the
# data. For example, you may want to have a subset containing only
# the most recent data, or a test set for specific kinds of inputs (e.g.,
# districts located inland versus districts located near the ocean).
# This will give you a deeper understanding of your model’s
# strengths and weaknesses.
# :::
# 
# [^footnote1]: A captcha is a test to ensure a user is not a robot. These tests have often been used as a cheap way to label
# training data.
# 
# 
# ### Backups
# 
# Finally, make sure you keep backups of every model you create and have the process
# and tools in place to roll back to a previous model quickly, in case the new model
# starts failing badly for some reason. Having backups also makes it possible to easily
# compare new models with previous ones. Similarly, you should keep backups of every
# version of your datasets so that you can roll back to a previous dataset if the new one
# ever gets corrupted (e.g., if the fresh data that gets added to it turns out to be full of
# outliers). Having backups of your datasets also allows you to evaluate any model
# against any previous dataset.
# 

# ## Conclusion
# 
# Much of the work is in the data preparation step: building monitoring
# tools, setting up human evaluation pipelines, and automating regular model training.
# ML algorithms are important, of course, but it is probably preferable to be comfortable with the overall process and know three or four algorithms well
# rather than to spend all your time exploring advanced algorithms.

# In[ ]:




