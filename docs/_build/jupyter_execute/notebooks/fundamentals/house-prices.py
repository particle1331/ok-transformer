#!/usr/bin/env python
# coding: utf-8

# # Pipelines in `scikit-learn`

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

# Pipelines in the [`scikit-learn` API](https://scikit-learn.org/stable/modules/classes.html#) allow us to sequentially apply a list of transformers and a final estimator on our dataset. Here, each intermediate step implements a `fit` and `transform` method, while the final estimator implements a `fit` method. Observe that this setup allows us to recursively fit of all transformations on the dataset in a single step. Moreover, these steps can be cross-validated together while setting different parameters. Another advantage of having pipelines is that it allows us to organize our implementation by separating declarative from imperative code making our code easier to read, debug, and maintain. 
# 
# ```{figure} ../../img/pipelines.png
# ---
# width: 40em
# name: pipelines
# ---
# Pipelines allow us to cleanly separate declarative from imperative code. [[source]](https://gh.mltrainings.ru/presentations/LopuhinJankiewicz_KaggleMercari.pdf)
# ```

# ## House prices dataset
# 
# Our task is to use California census data to build a model of housing prices. This data includes metrics such as the population, median income, and median housing price for each district in California. We will use mean absolute percentage error (MAPE) as our evaluation metric.

# ### Downloading the dataset

# ```{margin}
# ⚠️ **Attribution:** To demonstrate pipelines on an actual use case, we follow the [Chapter 2 notebook](https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb) of {cite}`geron2019hands-on` released under [Apache License 2.0](https://github.com/ageron/handson-ml2/blob/master/LICENSE). But we will go a bit deeper in exploring the capabilities of pipelines for feature engineering and hyperparameter optimization. 
# ```
# 

# Here we download the data using the `request.urlretrieve` function from `urllib`. This function takes a URL where the data is hosted and a save path where the data will be stored on the local disk.

# In[11]:


import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import tarfile
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


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


# Downloading...

# In[12]:


# Dataset URL
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Local save path
DATA_DIR = Path().resolve().parents[1] / 'data'
HOUSING_PATH = DATA_DIR / "housing"

# Downloading the data
fetch_housing_data(HOUSING_URL, HOUSING_PATH)


# ### Quick look at the data

# Let us load the data using pandas.

# In[13]:


housing = pd.read_csv(HOUSING_PATH / "housing.csv")
housing.head()


# In[14]:


housing.shape


# In[15]:


housing.info()


# The feature `total_bedrooms` is sometimes missing. All features are numerical except `ocean_proximity` which is text.

# In[16]:


housing.ocean_proximity.value_counts()


# The following plot shows how median house value is generally higher near the ocean. Moreover, prices seem to depend on population density.

# In[17]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c=housing["median_house_value"], cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
plt.grid();


# Let us look at the statistics of numerical features:

# In[18]:


housing.describe()


# We can better visualize this table using histograms.

# In[19]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# Distributions are tail-heavy; transforming the data to make it more bell-shaped may help some algorithms. Housing median age and house prices are capped.

# ### Getting a stratified test set

# Suppose we know that median income is a very
# important attribute to predict median housing prices. So we perform **stratified sampling** based on income categories to ensure that the test set is representative of the various categories of incomes in the whole dataset. 

# We check the percentage error based on the `income_cat` distribution of the whole dataset.

# In[20]:


from sklearn.model_selection import train_test_split

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# (Test dist - all dist) / all dist
uniform_error = (test_set['income_cat'].value_counts(normalize=True) - housing["income_cat"].value_counts(normalize=True))
uniform_error / housing["income_cat"].value_counts(normalize=True)


# In[21]:


strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, random_state=42, stratify=housing['income_cat'])

# (Strat test dist - all dist) / all dist
strat_error = strat_test_set['income_cat'].value_counts(normalize=True) - housing["income_cat"].value_counts(normalize=True)
strat_error / housing["income_cat"].value_counts(normalize=True)


# This looks way better. Let's drop the temporary indicator feature `income_cat` from the train and test sets.

# In[22]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# ## Preprocessing pipeline

# In this section we create two pipelines for separate preprocessing of categorical and numerical data. These two pipelines will be combined in a single full preprocessing pipeline for the training data. Reverting back to the original stratified train set:

# In[23]:


housing = strat_train_set.drop("median_house_value", axis=1)
targets = strat_train_set["median_house_value"].copy()


# ### Cleaning missing data

# We separate processing of numerical and categorical variables, then perform median imputation on numerical features.

# In[24]:


from sklearn.impute import SimpleImputer

housing_num = housing[[f for f in housing.columns if f != 'ocean_proximity']]
housing_cat = housing[['ocean_proximity']]

# Fitting the imputer on numerical fetures
imputer = SimpleImputer(strategy='median')
imputer.fit(housing_num)

# Checking...
(imputer.statistics_ == housing_num.median().values).all()


# Finally, we check that there are no more null values in the datasets

# In[25]:


housing_num_tr = pd.DataFrame(imputer.transform(housing_num), 
    columns=housing_num.columns, index=housing_num.index)

# Checking for nans
print(housing_num_tr.isna().sum().sum())
print(housing_cat.isna().sum().sum())


# ### Encoding categorical features

# We perform **one-hot encoding** on the `ocean_proximity` feature. An alternative would be **ordinal encoding** whose order is based on the mean target value of the data conditioned on that categorical variable.

# In[26]:


from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_onehot = cat_encoder.fit_transform(housing_cat)

housing_cat_onehot # sparse


# In[27]:


housing_cat_onehot.toarray() # dense


# In[28]:


cat_encoder.categories_ # learned categories


# ### Custom transformers

# Although scikit-learn provides many useful transformers, we will need to write own for tasks such as custom cleanup operations or combining specific
# attributes. Custom transformers work seamlessly with existing scikit-learn functionalities, all we need to do is create a class and implement three methods: `fit()` that returns `self`, `transform()`, and `fit_transform()`.
# 
# We can get `fit_transform()` for free by simply adding `TransformerMixin` as a base class. If we add `BaseEstimator` as a base class, we will also get two extra methods, `get_params()` and `set_params()`, that will be useful for automatic hyperparameter tuning. (Make sure to have no `*args`, and `**kwargs` in the constructor.)

# In[29]:


from sklearn.base import BaseEstimator, TransformerMixin

# Column indices
rooms_, bedrooms_, population_, households_ = 3, 4, 5, 6 


class CombinedFeaturesAdder(BaseEstimator, TransformerMixin):
    """Transformer for adding feature combinations discussed above."""
    
    def __init__(self, add_bedrooms_per_room=True): # No *args, **kwargs (!)
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        rooms_per_household = X[:, rooms_] / X[:, households_]
        population_per_household = X[:, population_] / X[:, households_]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_] / X[:, rooms_]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In this example the transformer has one hyperparameter, `add_bedrooms_per_room`,
# set to `True` by default (it is often helpful to provide sensible defaults). This hyperparameter will allow us to easily find out whether adding this attribute helps the
# ML algorithms or not. For now, we set this to `False`.

# In[30]:


feature_adder = CombinedFeaturesAdder(add_bedrooms_per_room=False)
housing_extra_features = feature_adder.transform(housing.values)


# Checking the shapes, the `feature_adder` should add two columns to the feature set.

# In[31]:


housing.shape


# In[32]:


housing_extra_features.shape


# We also modify `OneHotEncoder` so it doesn't fit on a training fold which may lack all some labels in `ocean_proximity` resulting in errors. 

# In[33]:


class OneHotEncoderModified(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer = OneHotEncoder()

    def fit(self, X, y=None):
        return self.transformer.fit(housing[['ocean_proximity']])
    
    def transform(self, X):
        return self.transformer.transform(X)


# ### Arranging the pipeline

# The `Pipeline` constructor takes a list of name estimator pairs defining a sequence of steps. All but the last estimator must be transformers (i.e. they must have a `fit_transform()` method). The names can be anything as long as they are unique and don't contain double underscores. Two things to note:
# 
# * Calling the pipeline's `fit()` method calls `fit_transform()` sequentially on all transformers, passing the output of each call as the parameter to the next call until it reaches the final estimator, for which it calls the `fit()` method.
# 
# +++
# 
# * The pipeline exposes the same methods as the final estimator. For our preprocessing pipeline, the last estimator is a `StandardScaler` is a transformer, so the pipeline has a `transform` method that applies all the transforms to the data in sequence.
# 
# Observe that we define two pipelines which transforms numerical and categorical features separately. These are combined using `ColumnTransformer` into a single pipeline. Note that `ColumnTransformer` sets `remainder='drop'` as the default which means non-specified columns in the list of transformers are dropped. We can specify `remainder='passthrough'` so that all remaining columns are skipped and  concatenated with the output of the transformers.

# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# Pipeline for numerical features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_adder', CombinedFeaturesAdder()),
    ('std_scaler', StandardScaler()),    
])

# Pipeline for categorical features
cat_pipeline = Pipeline([('one_hot', OneHotEncoderModified())])

# Combined full preprocessing pipeline
num_attribs = [col for col in housing.columns if col != "ocean_proximity"]
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# Preprocessed dataset
housing_transformed = full_pipeline.fit_transform(housing)
housing_transformed.shape


# `OneHotEncoder` returns a sparse matrix, while the `num_pipeline` returns
# a dense matrix. When there is such a mix of sparse and dense matrices, the `ColumnTransformer` estimates the density of the final matrix (i.e., the ratio of nonzero
# cells), and it returns a sparse matrix if the density is lower than a given threshold (by
# default, `sparse_threshold=0.3`). In this example, it returns a dense matrix.

# In[35]:


housing_transformed[:1]


# In[36]:


full_pipeline.sparse_threshold


# ## Fine tuning the pipeline

# Instead of tuning a single model, we will use `GridSearchCV` and `RandomSearchCV` can be used to tune the entire prediction pipeline.

# ### Prediction Pipeline = Preprocessing + RF

# The naming convention for the parameters to tune `param_grid` is a straightforward, recursive use of `__`  combining the names of the pipeline elements. This explains why we are not allowed to use double underscores and non-unique names for names of pipeline elements. 

# In[37]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Append estimator to preprocessing pipeline
model = RandomForestRegressor()
prediction_pipeline = Pipeline([
    ("preprocessing", full_pipeline),
    ("rf", model)
])

# Specify search space
param_grid = [{
    'preprocessing__num__feature_adder__add_bedrooms_per_room': [True, False], 
    'rf__n_estimators': range(200, 1000, 100),
    'rf__max_depth': range(5, 35, 5),
    }]

# Hyperparameter search
pipeline_tuner = RandomizedSearchCV(prediction_pipeline, 
                                    param_grid,
                                    n_jobs=-1,
                                    cv=5, 
                                    n_iter=20,
                                    scoring='neg_mean_squared_error') # for scoring, higher is better

# Fit unprocessed data
pipeline_tuner.fit(housing, targets)
print(pipeline_tuner.best_estimator_)


# Observe that `add_bedrooms_per_room=True` is the better parameter setting as it is the default setting in `CombinedFeaturesAdder`. This example shows that even preprocessing steps can be optimized as part of scikit-learn's pipeline architecture. Neat!

# In[38]:


pipeline_tuner.best_params_


# In[39]:


-pipeline_tuner.best_score_ # lower MSE


# In[40]:


results = pd.DataFrame(pipeline_tuner.cv_results_)
results['mean_test_score'] = -results['mean_test_score'] / 1e9 # lower is better
results['std_test_score'] = results['std_test_score'] / 1e9    # cv=5, lower is better
results.rename(columns={
    "param_preprocessing__num__feature_adder__add_bedrooms_per_room": "add_bpr",
    "param_rf__n_estimators": "n_estimators",
    "param_rf__max_depth": "max_depth"}, inplace=True)

remove_cols = ["params"] + [c for c in results.columns if ("split" in c) or ("time" in c)]
results.drop(remove_cols, axis=1, inplace=True)
results.sort_values("rank_test_score")


# In[54]:


from typing import Tuple

plt.figure(figsize=(9, 8))
results_bpr_true  = results.query("add_bpr==True")
results_bpr_false = results.query("add_bpr==False")
plt.scatter(results_bpr_true.n_estimators, results_bpr_true.max_depth, cmap=plt.get_cmap("jet"), c=results_bpr_true.mean_test_score, marker='o')
plt.scatter(results_bpr_false.n_estimators, results_bpr_false.max_depth, cmap=plt.get_cmap("jet"), c=results_bpr_false.mean_test_score, marker='x')
plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.colorbar()
plt.title("MSE (lower is better).\nMarkers: o = with BPR feature, x = without");

def annotate_circle(point: Tuple[int, int], annotation: str):
    circle_rad = 15  # This is the radius, in points
    plt.plot(point[0], point[1], 'o', ms=circle_rad*1.5, mec='k', mfc='none', mew=2)
    plt.annotate(annotation, xy=point, xytext=(-20, -20),
                textcoords='offset points',
                color='k', size='large')

# Plot circles around top 3 points
point = lambda r: list(results.sort_values("rank_test_score").iloc[r, :2].values)
annotate_circle(point(0), "1")
annotate_circle(point(1), "2")
annotate_circle(point(2), "3")


# Let's calculate the RMSE differences from the top parameters:

# In[42]:


scores = pd.DataFrame()
scores['RMSE'] = np.sqrt(results.sort_values("mean_test_score").mean_test_score.iloc[:5] * 1e9)
scores['Gap'] = scores.RMSE - scores.RMSE.shift(1)
scores.reset_index(drop=True)


# Results are really close. So we can probably choose the second best score with less, and shallower, trees.

# ### Feature importances in a pipeline?

# Getting feature importances is unfortunately very hacky... We have to manually reconstruct the features.

# In[43]:


# Random forest feature importances
best_pipeline = pipeline_tuner.best_estimator_
best_model = best_pipeline['rf']
feature_importances = best_model.feature_importances_

num_pipeline = best_pipeline['preprocessing'].named_transformers_['num']
cat_pipeline = best_pipeline['preprocessing'].named_transformers_['cat']

# List of numerical and categorical features
num_features = num_attribs + ['rooms_per_household', 'population_per_household']
num_features += ['bedrooms_per_room'] if num_pipeline['feature_adder'].add_bedrooms_per_room else []
cat_ohe_features = list(*cat_pipeline['one_hot'].transformer.categories_)

# Combine everything to get all columns
attributes = num_features + cat_ohe_features
feature_importance_df = pd.DataFrame(feature_importances, index=attributes, columns=['importance'])
feature_importance_df.sort_values('importance').plot.bar(figsize=(12, 6));


# ### Evaluating on the test set

# Note that the best pipeline obtained above will be used for inference without any modifications! Let's look at the train and test RMSEs:

# In[44]:


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

y_train = strat_train_set['median_house_value']
y_pred = best_pipeline.predict(strat_train_set.drop('median_house_value', axis=1))

print('Train RMSE:', np.sqrt(mean_squared_error(y_train, y_pred)))
print('Train MAPE:', mean_absolute_percentage_error(y_train, y_pred))


# In[45]:


y_test = strat_test_set['median_house_value']
y_pred = best_pipeline.predict(strat_test_set.drop('median_house_value', axis=1))

print('Test RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Test MAPE:', mean_absolute_percentage_error(y_test, y_pred))


# In[46]:


plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.4);
plt.plot(range(0, 500000, 10000), range(0, 500000, 10000), 'k--')
plt.xlabel('targets')
plt.ylabel('predictions');


# Let's try to compute for prices that were not clipped:

# In[47]:


FILTER = y_test < 400_000
y_test_small = y_test[FILTER]
X_test_small = strat_test_set.loc[FILTER, :].drop('median_house_value', axis=1)
y_pred_small = best_pipeline.predict(X_test_small)

print('RMSE:', np.sqrt(mean_squared_error(y_test_small, y_pred_small)))
print('MAPE:', mean_absolute_percentage_error(y_test_small, y_pred_small))


# Better RMSE! MAPE is worse since we have smaller denominators in this subset. The following results show that the model seem to generalize better on the filtered subset, i.e. for districts with price value less than $400,000. 

# In[48]:


from scipy import stats

def compute_confidence_interval(y_true, y_pred, confidence=0.95):
    """Calculating 95% CI for the generalization error using `scipy.stats.t.interval`."""

    squared_errors = (y_true - y_pred)**2
    m = len(squared_errors)
    ci = np.sqrt(stats.t.interval(
        confidence, 
        m - 1, 
        loc=squared_errors.mean(), 
        scale=stats.sem(squared_errors))
    )
    return ci


# Calculate the confidence interval (CI) for the test predictions. Observe that the CI is narrower for districts with lower true value.

# In[49]:


ci = compute_confidence_interval(y_test, y_pred)
print(list(ci))
print(ci[1] - ci[0])


# In[50]:


ci_small = compute_confidence_interval(y_test_small, y_pred_small)
print(list(ci_small))
print(ci_small[1] - ci_small[0])


# ## Conclusion
# 
# Pipelines are **awesome**. We can define custom transformers using mixins, which we can then use as part of scikit-learn pipelines, which makes this technique very flexible. For example, even preprocessing can be included in a pipeline, and therefore be part of cross-validation.

# 
