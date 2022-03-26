#!/usr/bin/env python
# coding: utf-8

# # Packaging Production Code

# ![Status](https://img.shields.io/static/v1.svg?label=Status&message=Ongoing&color=orange)

# Production code is designed to be deployed to end users as opposed to research code, which
# is for experimentation, building proof of concepts. Moreover, research code tends to be more short term in nature. On the other hand, with production code, we have some new considerations:
# 
# - **Testability and maintainability** are huge.
# We want to divide up our code into modules which are more extensible and easier to test.
# We separate config from code where possible, and ensure that functionality is tested and documented. We also look to ensure that our code adheres to standards like PEP 8 so that it's easy for others to read and maintain.  
# 
# +++
# 
# - **Scalability and performance** are also important areas to consider.
# With our production code, the code needs to be ready to be deployed to infrastructure that can be scaled. And in modern web applications, this typically means containerisation for vertical or horizontal scaling. 
# Where appropriate, we might also refactor inefficient parts of the code base.  
# 
# +++
# 
# - **Finally, we have to look at reproducibility**.
# The code resides under version control with clear processes for tracking releases and release versions, requirements, files, mark which dependencies and which versions are used by the code.
# 
# ```{margin}
# A **module** is basically just a Python file and a **package** is a
# collection of modules.
# ```
# 
# That is a quick overview of some of the key considerations with production code. In this article, we will be packaging up our machine learning model into a Python **package**. A package has certain standardized files which have to be present so that it can be published and then installed in other Python applications.
# Packaging allows us to wrap our train model and make it available to other consuming applications as a dependency, but with the additional benefits of version control, clear metadata and reproducibility.
# Note that [PyPI distributions](https://pypi.org/) have a 60MB limit after compression, so large models can't be published there. [This article](https://www.dampfkraft.com/code/distributing-large-files-with-pypi.html) provides multiple ways on how to overcome this size limitation for distributing Python packages.

# ## Code overview

# In order to create a package, we'll have to follow certain Python standards and conventions and
# we'll go into those in detail in this section. The structure of the resulting package looks like this.
# 
# ```
# .
# ├── regression_model
# │   ├── config
# │   ├── datasets
# │   ├── processing
# │   │   ├── __init__.py
# │   │   ├── data_manager.py
# │   │   ├── features.py
# │   │   └── validation.py
# │   ├── trained_models
# │   ├── pipeline.py
# │   ├── predict.py
# │   ├── train_pipeline.py
# │   ├── config.yml
# │   └── VERSION
# ├── requirements
# │   ├── requirements.txt
# │   └── test_requirements.txt
# ├── tests
# │   ├── conftest.py
# │   ├── test_features.py
# │   └── test_prediction.py
# ├── MANIFEST.in
# ├── mypy.ini
# ├── pyproject.toml
# ├── setup.py
# └── tox.ini
# ```

# `MANIFEST.in`, `pyproject.toml`, `setup.py`, `mypy.ini` and `tox.ini` are configurations either for packaging or for tooling like linting and type checking. We will be coming back to discuss these in more detail below. We have a `requirements/` directory, which is where we formalize the dependencies for our package and also for testing it. And we have a couple of sample tests in the `tests/` directory.  
# 
# The `regression_model/` directory is where the majority of our functionality is located. In this directory, we have three key files `train_pipeline.py`, `predict.py` and `pipeline.py`.
# These are sort of top level files for the key bits of functionality of the package. The `processing/` directory contains different helper functions. We have the datasets that we need to train and test the models in the `datasets/` directory. The `trained_models/` directory is where we save the models that we're persisting
# here as a pickle file so that it can be loaded in and accessed in the future. Finally, `config/` contains the core configurations module which reads the `config.yml` file.

# ## Package requirements

# Note that the `requirements/` directory has two requirements files. One for development, and one for the regression model. The versions listed in these files all adhere to [semantic versioning](https://www.geeksforgeeks.org/introduction-semantic-versioning/). Ranges are specified instead of exact versions since we assume that a minor version increment will not break the API. So what we've done in our requirements file is play it quite conservatively, taking advantage of bug fixes but also risking breaking the code in case the developers do not adhere to semantic versioning.
# 
# 
# ```{margin}
# [`requirements/requirements.txt`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/requirements/requirements.txt)
# ```
# 
# 
# ```
# numpy>=1.22.0,<1.23.0
# pandas>=1.4.0,<1.5.0
# pydantic>=1.8.1,<1.9.0
# scikit-learn>=1.0.0,<1.1.0
# strictyaml>=1.3.2,<1.4.0
# ruamel.yaml==0.16.12
# feature-engine>=1.0.2,<1.1.0
# joblib>=1.0.1,<1.1.0
# ```
# 
# The additional packages in the test requirements are only required when we want to test our package, or when we want to run style checks, linting and type checks:
# 
# ```{margin}
# [`requirements/test_requirements.txt`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/requirements/test_requirements.txt)
# ```
# 
# ```
# # Install requirements.txt along with others
# -r requirements.txt
# 
# # Testing requirements
# pytest>=6.2.3,<6.3.0
# 
# # Repo maintenance tooling
# black==20.8b1
# flake8>=3.9.0,<3.10.0
# mypy==0.812
# isort==5.8.0
# ```
# 
# This `requirements.txt` approach to managing our projects dependencies is probably the most basic way of doing dependency management in Python.
# Nothing wrong with it at all. Many of the biggest python open source projects out there use this exact approach. There are other dependency managers out there such as Poetry and Pipenv. But the principle of defining your dependencies and specifying the version ranges remains the same across all of the tools.

# ## Working with tox

# 
# Now what we're going to do is see our package in action on some of its main commands.
# To start, if we've just cloned the repository and we have a look at our `trained_models/` directory you can see that its empty. There are no other files inside train models right now. We can generate a trained model serialized as a `.pkl` file by running:
# 
# ```
# tox -e train
# ``` 
# 
# Here we've used `tox` to trigger our train pipeline script. So what's `tox`? How does it work? `tox` is a generic virtual environment management and test command line tool. For our purposes here, this means that with `tox`, we don't have to worry about different operating systems. We don't have to worry about things like setting up Python paths, configuring environment variables. We do all of that stuff inside our `tox.ini` file. 
# This is a great tool, and it's worth adding to your toolbox to get started with tox.
# 

# ```{margin}
# [`tox.ini`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/tox.ini)
# ```
# 
# ```ini
# # Tox is a generic virtualenv management and test command line tool. Its goal is to
# # standardize testing in Python. We will be using it extensively in this course.
# 
# # Using Tox we can (on multiple operating systems):
# # + Eliminate PYTHONPATH challenges when running scripts/tests
# # + Eliminate virtualenv setup confusion
# # + Streamline steps such as model training, model publishing
# 
# 
# [tox]
# envlist = test_package, typechecks, stylechecks, lint
# skipsdist = True
# 
# 
# [testenv]
# install_command = pip install {opts} {packages}
# 
# 
# [testenv:test_package]
# deps =
# 	-rrequirements/test_requirements.txt
# 
# setenv =
# 	PYTHONPATH=.
# 	PYTHONHASHSEED=0
# 
# commands =
# 	python regression_model/train_pipeline.py
# 	pytest -s -v tests/
# 
# 
# [testenv:train]
# envdir 	 = {toxworkdir}/test_package
# deps 	 = {[testenv:test_package]deps}
# setenv 	 = {[testenv:test_package]setenv}
# commands = 
# 	python regression_model/train_pipeline.py
# 
# # ...
# ```

# Every time you see something in square brackets like this, this is a different tox environment and an environment is something which is going to set up a virtual environment in your `.tox` hidden directory. We can run commands within a specific environment, and we can also inherit commands and dependencies from other environments. This is a sort of foundational unit when we're working with tox.

# Here, we have the default `tox` environment and a default `testenv` environment.
# And what this means is that if we just run the `tox` command on its own, it's going to run all the commands in these different environments (`test_package`, `typechecks`, `stylechecks`, and `lint`). You will see that these names corresponds to environments defined further in the file. Continuing, we set `skipsdist=True` since we don't want to build the package when using tox. The `testenv` is almost like a base class, if you think of inheritance (using the `:` syntax). And so this `install_command` is going to be consistent whenever we inherit from this base environment.

# For `test_package` environment which inherits from `testenv`, we define `deps` and that tells `tox` that for this particular environment, we're going to need to install `requirements/test_requirements.txt` with flag `-r`. This also sets environmental variables `PYTHONPATH=.` for the root directory and `PYTHONHASHSEED=0` to disable setting hash seed to a random integer for test commands. Finally, the following two commands are run:
# 
# ```
# $ python regression_model/train_pipeline.py
# $ pytest -s -v tests
# ```
# 
# Here `-s` means to disable all capturing and `-v` to get verbose outputs. You can test this by running the following script in the terminal:
# 
# ```
# $ tox -e test_package
# test_package installed: appdirs==1.4.4,attrs==21.4.0,black==20.8b1,click==8.0.4,feature-engine==1.0.2,flake8==3.9.2,iniconfig==1.1.1,isort==5.8.0,joblib==1.0.1,mccabe==0.6.1,mypy==0.812,mypy-extensions==0.4.3,numpy==1.22.3,packaging==21.3,pandas==1.4.1,pathspec==0.9.0,patsy==0.5.2,pluggy==1.0.0,py==1.11.0,pycodestyle==2.7.0,pydantic==1.8.2,pyflakes==2.3.1,pyparsing==3.0.7,pytest==6.2.5,python-dateutil==2.8.2,pytz==2021.3,regex==2022.3.2,ruamel.yaml==0.16.12,ruamel.yaml.clib==0.2.6,scikit-learn==1.0.2,scipy==1.8.0,six==1.16.0,statsmodels==0.13.2,strictyaml==1.3.2,threadpoolctl==3.1.0,toml==0.10.2,typed-ast==1.4.3,typing_extensions==4.1.1
# test_package run-test-pre: PYTHONHASHSEED='0'
# test_package run-test: commands[0] | python regression_model/train_pipeline.py
# test_package run-test: commands[1] | pytest -s -v tests/
# ============================= test session starts ==============================
# platform darwin -- Python 3.8.12, pytest-6.2.5, py-1.11.0, pluggy-1.0.0 -- /Users/particle1331/code/model-deployment/packages/regression_model.tox/test_package/bin/python
# cachedir: .tox/test_package/.pytest_cache
# rootdir: /Users/particle1331/code/model-deployment/production, configfile: pyproject.toml
# collected 2 items
# 
# tests/test_features.py::test_temporal_variable_transformer PASSED
# tests/test_prediction.py::test_make_prediction PASSED
# 
# ============================== 2 passed in 0.19s ===============================
# ___________________________________ summary ____________________________________
#   test_package: commands succeeded
#   congratulations :)
# ```

# Next, we have the `train` environment. Notice that we set `envdir={toxworkdir}/test_package` which tells tox to recreate the `test_package` environment in the hidden `.tox` directory. This is to save time as setting up a new virtual environment takes a while. Furthermore, setting `deps={[testenv:test_package]deps}` installs `test_requirements.txt` instead of `requirements.txt`. Again this saves time, though the train script should not require tooling libraries. After setting up the environment, the training pipeline is triggered without running the tests:

# ```
# $ tox -e train
# train installed: appdirs==1.4.4,attrs==21.4.0,black==20.8b1,click==8.0.4,feature-engine==1.0.2,flake8==3.9.2,iniconfig==1.1.1,isort==5.8.0,joblib==1.0.1,mccabe==0.6.1,mypy==0.812,mypy-extensions==0.4.3,numpy==1.22.3,packaging==21.3,pandas==1.4.1,pathspec==0.9.0,patsy==0.5.2,pluggy==1.0.0,py==1.11.0,pycodestyle==2.7.0,pydantic==1.8.2,pyflakes==2.3.1,pyparsing==3.0.7,pytest==6.2.5,python-dateutil==2.8.2,pytz==2021.3,regex==2022.3.2,ruamel.yaml==0.16.12,ruamel.yaml.clib==0.2.6,scikit-learn==1.0.2,scipy==1.8.0,six==1.16.0,statsmodels==0.13.2,strictyaml==1.3.2,threadpoolctl==3.1.0,toml==0.10.2,typed-ast==1.4.3,typing_extensions==4.1.1
# train run-test-pre: PYTHONHASHSEED='0'
# train run-test: commands[0] | python regression_model/train_pipeline.py
# ___________________________________ summary ____________________________________
#   train: commands succeeded
#   congratulations :)
# ```

# If you look at the `tox.ini` source file, we also have tox commands for running our type checks, style checks, and linting. These are defined following the same pattern as the `train` environment.

# ## Package config

# In this section, we are going to talk about how we structure our config. You may have noticed that we have a `config.yml` file here inside the `regression_model/` directory. A good rule of thumb is that you want to limit the amount of power that your config files have. If you write them in Python, it'll be tempting to add small bits of Python code and that can cause bugs. Moreover, config files in standard formats like YAML or JSON can also be edited by developers who don't know Python. For our purposes, we have taken all those global constants and hyperparameters, and put them in YAML format in the `config.yml` file.
# 

# ```{margin}
# [`regression_model/config.yml`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/config.yml)
# ```
# 
# ````{note} 
# If you're not familiar with YAML syntax, we explain its most relevant features here. Key-value pairs corresponds to an assignment operation: `package_name: regression_model` will be loaded as `package_name = "regression_model"` in Python. Nested keys with indentation will be read as keys of a dictionary:
# 
# ```yaml
# variables_to_rename:
#   1stFlrSF: FirstFlrSF
#   2ndFlrSF: SecondFlrSF
#   3SsnPorch: ThreeSsnPortch
# ```
# 
# ```python
# variables_to_rename = {'1stFlrSF': 'FirstFlrSF', '2ndFlrSF': 'SecondFlrSF', '3SsnPorch': 'ThreeSsnPortch'}
# ```
# 
# Finally, we have the indented hyphen syntax which is going to be a list.
# 
# ```yaml
# numericals_log_vars:
#   - LotFrontage
#   - FirstFlrSF
#   - GrLivArea
# ```
# 
# ```python
# numericals_log_vars = ['LotFrontage', 'FirstFlrSF', 'GrLivArea']
# ```
# ````

# If we head over to the `config/` directory, we have our `core.py` file, there are a few things that are happening here. First, we are using `pathlib` to define the location of files and directories that we're interested in using. Here `regression_model.__file__` refers to the `__init__.py` file in `regression_model/`, so that `PACKAGE_ROOT` refers to the path of `regression_model/`. We also define the paths of the config YAML file, the datasets, and trained models.

# ```{margin}
# [`regression_model/config/core.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/config/core.py)
# ```
# 
# ```python
# # Project Directories
# PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
# ROOT = PACKAGE_ROOT.parent
# CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
# DATASET_DIR = PACKAGE_ROOT / "datasets"
# TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
# 
# 
# class AppConfig(BaseModel):
#     """
#     Application-level config.
#     """
# 
#     package_name: str
#     training_data_file: str
#     test_data_file: str
#     pipeline_save_file: str
# 
# 
# class ModelConfig(BaseModel):
#     """
#     All configuration relevant to model training and feature engineering.
#     """
# 
#     target: str
#     variables_to_rename: Dict
#     features: List[str]
#     test_size: float
#     random_state: int
#     alpha: float
#     categorical_vars_with_na_frequent: List[str]
#     categorical_vars_with_na_missing: List[str]
#     numerical_vars_with_na: List[str]
#     temporal_vars: List[str]
#     ref_var: str
#     numericals_log_vars: Sequence[str]
#     binarize_vars: Sequence[str]
#     qual_vars: List[str]
#     exposure_vars: List[str]
#     finish_vars: List[str]
#     garage_vars: List[str]
#     categorical_vars: Sequence[str]
#     qual_mappings: Dict[str, int]
#     exposure_mappings: Dict[str, int]
#     garage_mappings: Dict[str, int]
#     finish_mappings: Dict[str, int]
# ```

# Here we use `BaseModel` from `pydantic` to define our config classes. 
# Pydantic is an excellent library for data validation and settings management using Python type annotations. This is really powerful because it means we don't have to learn a new sort of micro language for data parsing and schema validation.
# We can just use Pydantic and our existing knowledge of Python type hints.
# And so, this gives us a really clear and powerful way to understand and 
# potentially test our config, and to prevent introducing bugs into our model.
# 
# For the sake of separating concerns, we define two subconfigs: everything to do with our 
# model, and then everything to do with our package. Developmental concerns, like the package name and 
# the location of the pipeline, go into the `AppConfig` data model. The data science configs
# go into `ModelConfig`. Then, we wrap it in an overall config:

# ```{margin}
# [`regression_model/config/core.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/config/core.py)
# ```
# 
# ```python
# class Config(BaseModel):
#     """Master config object."""
# 
#     app_config: AppConfig
#     model_config: ModelConfig
# ```
# 
# At the bottom of the `core` config module, we have three helper functions. Our `config` object, 
# which is what we're going to be importing in other modules, is defined through this 
# `create_and_validate_config` function.
# This uses our `parse_config_from_yaml` function, which using `CONFIG_FILE_PATH` specified above
# will check that the file exists, and then attempt to load it using the `strictyaml` load function.
# 
# 
# ```{margin}
# [`regression_model/config/core.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/config/core.py)
# ```
# 
# ```python
# def validate_config_file_path(cfg_path: Path) -> Path:
#     """Locate the configuration file."""
# 
#     if not cfg_path.is_file():
#         raise OSError(f"Config not found at {cfg_path!r}")
# 
#     return cfg_path
# 
# 
# def parse_config_from_yaml(cfg_path: Path) -> YAML:
#     """Parse YAML containing the package configuration."""
# 
#     cfg_path = validate_config_file_path(cfg_path)
#     with open(cfg_path, "r") as conf_file:
#         parsed_config = load(conf_file.read())
# 
#     return parsed_config
# 
# 
# def create_and_validate_config(parsed_config: YAML) -> Config:
#     """Run validation on config values."""
# 
#     return Config(
#         app_config=AppConfig(**parsed_config.data),
#         model_config=ModelConfig(**parsed_config.data),
#     )
# 
# 
# _parsed_config = parse_config_from_yaml(CONFIG_FILE_PATH)
# config = create_and_validate_config(_parsed_config)
# ```

# And once we load it in our YAML file, we then unpack the key value
# pairs here and pass them to `AppConfig` and `ModelConfig` as keyword arguments 
# to instantiate these classes.
# And that results in us having this `config` object, which is what we are going to be importing around our package.

# ## Model training pipeline

# Now that we've looked at our config, let's dig into the main `regression/train_pipeline.py` scripts. This is what we've been running in our tox commands.
# If we open up this file, you can see we have one function, which is `run_training`.
# And if we step through what's happening here, we are loading in the training data and we've created
# some utility functions like this `load_dataset` function, which comes from our `data_manager/` module. After loading, we use the standard scikit `train_test_split`. The test set obtained here can be used to evaluate the model (which can be part of the automated tests). Here we are making use of our config object to specify the parameters of this function. 
# It's important to note that we log-transform our targets prior to training.
# 
# ```{margin}
# [`regression_model/train_pipeline.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/train_pipeline.py) 
# ```
# 
# ```python
# import numpy as np
# from config.core import config
# from pipeline import price_pipe
# from processing.data_manager import load_dataset, save_pipeline
# from sklearn.model_selection import train_test_split
# 
# 
# def run_training() -> None:
#     """Train the model."""
# 
#     # Read training data
#     data = load_dataset(file_name=config.app_config.training_data_file)
# 
#     # Divide train and test
#     X_train, X_test, y_train, y_test = train_test_split(
#         data[config.model_config.features],
#         data[config.model_config.target],
#         test_size=config.model_config.test_size,
#         random_state=config.model_config.random_state,
#     )
#     y_train = np.log(y_train)   # <-- ⚠ Invert before serving preds
# 
#     # Fit model
#     price_pipe.fit(X_train, y_train)
# 
#     # Persist trained model
#     save_pipeline(pipeline_to_persist=price_pipe)
# 
# 
# if __name__ == "__main__":
#     run_training()
# ```
# 

# The load function is defined as follows. In the `clean_inputs` helper function, we perform some basic preprocessing such as converting `MSSubClass` to type `object` since its a categorical variable (see `config.yml`), although the values are numeric.
# We also rename variables beginning with numbers in to avoid syntax errors. The `*` syntax forces zero positional arguments so that all arguments are named when passed.  These are technical fixes that should not affect the quality of the model.

# ```{margin}
# [`regression_model/processing/data_manager.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/processing/data_manager.py)
# ```
# 
# ```python
# def clean_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
#     """Clean data to avoid syntax errors later."""
#     
#     input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
#     input_data["MSSubClass"] = input_data["MSSubClass"].astype("O")
#     return input_data
# 
# 
# def load_dataset(*, file_name: str) -> pd.DataFrame:
#     """Load and clean dataset."""
#     
#     dataframe = pd.read_csv(DATASET_DIR / file_name)
#     dataframe = clean_inputs(input_data=dataframe)
#     return dataframe
# ```

# Next, we have our `price_pipe` which is a `scikit-learn` pipeline object and we'll look at the `pipeline` module in a moment, in the next section. But you can see here how we use it to fit the data. After fitting the pipeline, we use the `save_pipeline` function to persist it. This also takes care of naming the pipeline which depends on the current package version. 
# The other nontrivial part of the save function is the `remove_old_pipelines` which deletes all files inside `trained_models/` so long as the file is not the init file. This ensures that there is always precisely one model inside the storage directory minimizing the chance of making a mistake.

# 
# ```{margin}
# [`regression_model/processing/data_manager.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/processing/data_manager.py)
# ```
# 
# ```python
# def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
#     """Persist the pipeline.
#     Saves the versioned model, and overwrites any previous saved models.
#     This ensures that when the package is published, there is only one
#     trained model that can be called, and we know exactly how it was built."""
#     
#     # Prepare versioned save file name
#     save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
#     save_path = TRAINED_MODEL_DIR / save_file_name
# 
#     remove_old_pipelines()
#     joblib.dump(pipeline_to_persist, save_path)
# 
# 
# def remove_old_pipelines() -> None:
#     """Remove old model pipelines.
#     This is to ensure there is a simple one-to-one mapping between 
#     the package version and the model version to be imported and 
#     used by other applications."""
#     
#     do_not_delete = ["__init__.py"]
#     for model_file in TRAINED_MODEL_DIR.iterdir():
#         if model_file.name not in do_not_delete:
#             model_file.unlink()  # Delete
# ```

# And then the last step in our save pipeline function is to use the job serialisation library to persist
# the pipeline to the save path that we've defined. And that's how our `regression_model_output_version_v0.0.1.pkl` ends up here in `trained_models/`.

# ## Feature engineering

#  In this section we will look at our feature engineering pipeline. Looking at the code, we're applying transformations sequentially to preprocess and feature engineer our data. Thanks to the `feature_engine` API each step is almost human readable, we only have to set the variables where we apply the transformations.
# 
# ```{margin}
# [`regression_model/pipeline.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/pipeline.py)
# ```
# 
# ```python
# from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
# from feature_engine.imputation import (
#     AddMissingIndicator,
#     CategoricalImputer,
#     MeanMedianImputer,
# )
# from feature_engine.selection import DropFeatures
# from feature_engine.transformation import LogTransformer
# from feature_engine.wrappers import SklearnTransformerWrapper
# from sklearn.linear_model import Lasso
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import Binarizer, MinMaxScaler
# 
# from regression_model.config.core import config
# from regression_model.processing import features as pp
# 
# price_pipe = Pipeline(
#     [
#         # ===== IMPUTATION =====
#         # Impute categorical variables with string missing
#         (
#             "missing_imputation",
#             CategoricalImputer(
#                 imputation_method="missing",
#                 variables=config.model_config.categorical_vars_with_na_missing,
#             ),
#         ),
#         # Impute categorical variables with most frequent category
#         (
#             "frequent_imputation",
#             CategoricalImputer(
#                 imputation_method="frequent",
#                 variables=config.model_config.categorical_vars_with_na_frequent,
#             ),
#         ),
#         # Add missing indicator
#         (
#             "missing_indicator",
#             AddMissingIndicator(variables=config.model_config.numerical_vars_with_na),
#         ),
#         # Impute numerical variables with the mean
#         (
#             "mean_imputation",
#             MeanMedianImputer(
#                 imputation_method="mean",
#                 variables=config.model_config.numerical_vars_with_na,
#             ),
#         ),
#         
#         # == TEMPORAL VARIABLES ====
#         (
#             "elapsed_time",
#             pp.TemporalVariableTransformer(
#                 variables=config.model_config.temporal_vars,
#                 reference_variable=config.model_config.ref_var,
#             ),
#         ),
#         ("drop_features", DropFeatures(features_to_drop=[config.model_config.ref_var])),
#         
#         # ==== VARIABLE TRANSFORMATION =====
#         ("log", LogTransformer(variables=config.model_config.numericals_log_vars)),
#         (
#             "binarizer",
#             SklearnTransformerWrapper(
#                 transformer=Binarizer(threshold=0),
#                 variables=config.model_config.binarize_vars,
#             ),
#         ),
#         
#         # === MAPPERS ===
#         (
#             "mapper_qual",
#             pp.Mapper(
#                 variables=config.model_config.qual_vars,
#                 mappings=config.model_config.qual_mappings,
#             ),
#         ),
#         (
#             "mapper_exposure",
#             pp.Mapper(
#                 variables=config.model_config.exposure_vars,
#                 mappings=config.model_config.exposure_mappings,
#             ),
#         ),
#         (
#             "mapper_finish",
#             pp.Mapper(
#                 variables=config.model_config.finish_vars,
#                 mappings=config.model_config.finish_mappings,
#             ),
#         ),
#         (
#             "mapper_garage",
#             pp.Mapper(
#                 variables=config.model_config.garage_vars,
#                 mappings=config.model_config.garage_mappings,
#             ),
#         ),
#         
#         # == CATEGORICAL ENCODING
#         # Encode infrequent categorical variable with category "Rare"
#         (
#             "rare_label_encoder",
#             RareLabelEncoder(
#                 tol=0.01, n_categories=1, variables=config.model_config.categorical_vars
#             ),
#         ),        
#         # Encode categorical variables using the target mean
#         (
#             "categorical_encoder",
#             OrdinalEncoder(
#                 encoding_method="ordered",
#                 variables=config.model_config.categorical_vars,
#             ),
#         ),
#         ("scaler", MinMaxScaler()),
# 
#         # == REGRESSION MODEL (LASSO)
#         (
#             "Lasso",
#             Lasso(
#                 alpha=config.model_config.alpha,
#                 random_state=config.model_config.random_state,
#             ),
#         ),
#     ]
# )
# ```

# Note that although we're using a lot of transformers from the `feature_engine` library, we also have some custom ones that we've created
# in the `processing.features` module of our package. First, we have `TemporalVariableTransformer` which inherits from `BaseEstimator` and `TransformerMixin` in `sklearn.base`. 
# By doing this, and also ensuring that we specify a `fit` and a `transform` method, we're able to use this to transform variables and it's compatible with our `scikit-learn` pipeline. 
# 
# The transformation defined in `TemporalVariableTransformer` replaces any temporal variable `t` with `t0 - t` for some reference variable `t0`. From expericen, intervals work better for linear models than specific values (such as years). Then, the variable `t0` is dropped since its information has been incorporated in the other temporal variables. 

# ```{margin}
# [`regression_model/processing/features.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/processing/features.py)
# ```
# ```python
# class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
#     """Temporal elapsed time transformer."""
# 
#     def __init__(self, variables: List[str], reference_variable: str):
# 
#         if not isinstance(variables, list):
#             raise ValueError("variables should be a list")
# 
#         self.variables = variables
#         self.reference_variable = reference_variable
# 
#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         return self
# 
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         # So that we do not over-write the original DataFrame
#         X = X.copy()
# 
#         for feature in self.variables:
#             X[feature] = X[self.reference_variable] - X[feature]
# 
#         return X
# ```

# Next, we have the `Mapper` class which simply maps features to other values as specified in the `mappings` dictionary argument. The mappings and the mapped variables are specified in the config file. 

# ```{margin}
# [`regression_model/processing/features.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/processing/features.py)
# ```
# ```python
# class Mapper(BaseEstimator, TransformerMixin):
#     """Categorical variable mapper."""
# 
#     def __init__(self, variables: List[str], mappings: dict):
# 
#         if not isinstance(variables, list):
#             raise ValueError("variables should be a list")
# 
#         self.variables = variables
#         self.mappings = mappings
# 
#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         return self
# 
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         X = X.copy()
#         for feature in self.variables:
#             X[feature] = X[feature].map(self.mappings)
# 
#         return X
# ```

# We could easily create additional feature engineering steps here by adding transformations that adhere to this structure (defining custom `sklearn` transformers), then adding
# it to our pipeline at whatever point in the pipeline it made sense, and specifying which variables the transformers apply to by implementing a `variables` attribute. Note that each step takes the whole output of the previous step as input which is why we implement this attribute. 

# ### Testing our feature transformation

# Here we show how you can test specific steps in the pipeline. In particular, those that are defined in the `processing.features` module. From the `test.csv` dataset, we can see in the first line that the `YrRemodAdd` is 1961 and `YrSold` is 2010. Thus, we expect the transformed `YrRemodAdd` value to be 49. This is reflected in the following test. Take note of the structure of the test where we specify the context, conditions, and expectations. 

# ```{margin}
# [`tests/test_features.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/tests/test_features.py)
# ```
# 
# ```python
# def test_temporal_variable_transformer(sample_input_data):
#     # Given
#     transformer = TemporalVariableTransformer(
#         variables=config.model_config.temporal_vars,  # YearRemodAdd
#         reference_variable=config.model_config.ref_var,
#     )
#     assert sample_input_data["YearRemodAdd"].iat[0] == 1961
# 
#     # When
#     subject = transformer.fit_transform(sample_input_data)
# 
#     # Then
#     assert subject["YearRemodAdd"].iat[0] == 49
# ```

# Note that fixture `sample_input_data` is the `test.csv` dataset loaded in the [`conftest` module](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/tests/conftest.py). Here replacing the test with any integer other than 49 will break the test. In an actual project, we should have unit tests here for every bit of feature engineering that we will do. As well as some more complex tests that go along with the spirit of the feature engineering and feature transformation pipeline. 

# ## Input validation and prediction pipeline

# For the final piece of functionality, we take a look at the prediction pipeline of our regression model. The concerned functions live in the `predict` and `validation` modules. We also use our load function in `data_manager/` which simply implements `joblib.load` to work with our package structure.

# ```{margin}
# [`regression_model/processing/data_manager.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/processing/data_manager.py)
# ```
# 
# ```python
# def load_pipeline(*, file_name: str) -> Pipeline:
#     """Load a persisted pipeline."""
#     
#     file_path = TRAINED_MODEL_DIR / file_name
#     trained_model = joblib.load(filename=file_path)
#     return trained_model
# ```

# Now let's look at the `make_prediction` function. This function expects a pandas `DataFrame` or a dictionary, validates the data, then makes a prediction only when the data is valid. Note that the return value has the same format for all branches of the function. Moreover, the transformation `np.exp` is applied on the model outputs since the model is trained to have outputs in logarithmic scale.

# ```{margin}
# [`regression_model/predict.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/predict.py)
# ```
# 
# 
# ```python
# pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
# _price_pipe = load_pipeline(file_name=pipeline_file_name)
# 
# 
# def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
#     """Make a prediction using a saved model pipeline."""
# 
#     data = pd.DataFrame(input_data)
#     validated_data, errors = validate_inputs(input_data=data)
#     results = {"predictions": None, "version": _version, "errors": errors}
# 
#     if not errors:
#         predictions = _price_pipe.predict(
#             X=validated_data[config.model_config.features]
#         )
#         results = {
#             "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
#             "version": _version,
#             "errors": errors,
#         }
# 
#     return results
# ```

# Testing the actual function:

# In[2]:


from pathlib import Path
import pandas as pd
from regression_model import datasets
from regression_model.processing.validation import *
from regression_model.config.core import config
from regression_model.predict import make_prediction

test = pd.read_csv(Path(datasets.__file__).resolve().parent/ "test.csv")
make_prediction(input_data=test.iloc[:5])


# The `make_prediction` function depends heavily on the `validate_inputs` function defined below.

# ```{margin}
# [`regression_model/processing/validation.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/processing/validation.py)
# ```
# 
# ```python
# def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
#     """Check model inputs for unprocessable values."""
# 
#     input_data = clean_inputs(input_data=input_data)
#     relevant_data = input_data[config.model_config.features].copy()
#     validated_data = drop_na_inputs(input_data=relevant_data)
#     errors = None
# 
#     try:
#         # Replace numpy nans with None so that pydantic can validate
#         MultipleHouseDataInputs(
#             inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
#         )
#     except ValidationError as error:
#         errors = error.json()
# 
#     return validated_data, errors
# ```

# First, this function loads and cleans the input dataset. Then, the function applies `drop_na_inputs` which is defined below. This function looks at the features that were never missing on any of the training examples, but for some reason are missing on some examples in the test set. In our implementation, we simply skip making predictions on such examples. (How to exactly handle this would depend on the actual use case.)

# ```{margin}
# [`regression_model/processing/validation.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/processing/validation.py)
# ```
# 
# ```python
# def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
#     """Check model inputs for na values and filter."""
#     
#     validated_data = input_data.copy()
#     new_vars_with_na = [
#         var
#         for var in config.model_config.features
#         if var
#         not in config.model_config.categorical_vars_with_na_frequent
#         + config.model_config.categorical_vars_with_na_missing
#         + config.model_config.numerical_vars_with_na
#         and validated_data[var].isnull().sum()
#         > 0  # at least one missing example has missing var
#     ]
#     validated_data.dropna(axis=0, subset=new_vars_with_na, inplace=True)  # drop rows
# 
#     return validated_data
# 

# Finally, `validate_inputs` uses the Pydantic model `HouseDataInputSchema` to check whether the input date have the expected types. Note, if you look at the source code, the input schema defines type for each possible feature &mdash; not just the selected features. This can be useful if ever we change the selected features. The `MultipleHouseDataInputs` is simply a wrapper so we can apply validation on a list of inputs. Also `.to_dict(orient="records")` means to transform the data to a list of dictionaries with column as keys.

# ```{margin}
# [`regression_model/processing/validation.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/regression_model/processing/validation.py)
# ```
# 
# ```python
# class HouseDataInputSchema(BaseModel):
#     Alley: Optional[str]
#     BedroomAbvGr: Optional[int]
#     BldgType: Optional[str]
#     BsmtCond: Optional[str]
#     BsmtExposure: Optional[str]
#     BsmtFinSF1: Optional[float]
#     BsmtFinSF2: Optional[float]
#     BsmtFinType1: Optional[str]
#     BsmtFinType2: Optional[str]
#     BsmtFullBath: Optional[float]
#     BsmtHalfBath: Optional[float]
#     BsmtQual: Optional[str]
#     BsmtUnfSF: Optional[float]
#     CentralAir: Optional[str]
#     Condition1: Optional[str]
#     Condition2: Optional[str]
#     Electrical: Optional[str]
#     EnclosedPorch: Optional[int]
#     ExterCond: Optional[str]
#     ExterQual: Optional[str]
#     Exterior1st: Optional[str]
#     Exterior2nd: Optional[str]
#     Fence: Optional[str]
#     FireplaceQu: Optional[str]
#     Fireplaces: Optional[int]
#     Foundation: Optional[str]
#     FullBath: Optional[int]
#     Functional: Optional[str]
#     GarageArea: Optional[float]
#     GarageCars: Optional[float]
#     GarageCond: Optional[str]
#     GarageFinish: Optional[str]
#     GarageQual: Optional[str]
#     GarageType: Optional[str]
#     GarageYrBlt: Optional[float]
#     GrLivArea: Optional[int]
#     HalfBath: Optional[int]
#     Heating: Optional[str]
#     HeatingQC: Optional[str]
#     HouseStyle: Optional[str]
#     Id: Optional[int]
#     KitchenAbvGr: Optional[int]
#     KitchenQual: Optional[str]
#     LandContour: Optional[str]
#     LandSlope: Optional[str]
#     LotArea: Optional[int]
#     LotConfig: Optional[str]
#     LotFrontage: Optional[float]
#     LotShape: Optional[str]
#     LowQualFinSF: Optional[int]
#     MSSubClass: Optional[int]
#     MSZoning: Optional[str]
#     MasVnrArea: Optional[float]
#     MasVnrType: Optional[str]
#     MiscFeature: Optional[str]
#     MiscVal: Optional[int]
#     MoSold: Optional[int]
#     Neighborhood: Optional[str]
#     OpenPorchSF: Optional[int]
#     OverallCond: Optional[int]
#     OverallQual: Optional[int]
#     PavedDrive: Optional[str]
#     PoolArea: Optional[int]
#     PoolQC: Optional[str]
#     RoofMatl: Optional[str]
#     RoofStyle: Optional[str]
#     SaleCondition: Optional[str]
#     SaleType: Optional[str]
#     ScreenPorch: Optional[int]
#     Street: Optional[str]
#     TotRmsAbvGrd: Optional[int]
#     TotalBsmtSF: Optional[float]
#     Utilities: Optional[str]
#     WoodDeckSF: Optional[int]
#     YearBuilt: Optional[int]
#     YearRemodAdd: Optional[int]
#     YrSold: Optional[int]
#     FirstFlrSF: Optional[int]  # renamed
#     SecondFlrSF: Optional[int]  # renamed
#     ThreeSsnPortch: Optional[int]  # renamed
# 
# 
# class MultipleHouseDataInputs(BaseModel):
#     inputs: List[HouseDataInputSchema]
# ```

# The `validate_inputs` function only uses the input schema to retrieve errors, the data it returns is still the filtered output `drop_na_inputs`. This makes sense because has pandas already inferred the types, and we are only confirming whether it has inferred the correct types. Finally, note that even a single type error for one row (here out of more than a thousand) would make the prediction function fail. This behavior can be devastating in a production environment if not expected by the devs.

# ### Testing predictions

# Let us try to make predictions on the first 5 test examples. Then, we print the number of expected predictions from the test set. Note that this also checks the validity of our test data (not just the regression model).

# In[7]:


result = make_prediction(input_data=test)
predictions = result.get("predictions")
print('First 5 predictions:\n', predictions[:5])
print('Expected no. of predictions:\n', validate_inputs(input_data=test)[0].shape[0])


# These facts make up our tests for the prediction pipeline. Again the fixture `sample_input_data` is actually `test.csv` loaded in the [`conftest` module](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/tests/conftest.py). Recall that with the train-test split in `run_training`, we can add the validation performance of the trained model as part of automated tests. 

# ```{margin}
# [`tests/test_prediction.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/tests/test_prediction.py)
# ```
# 
# ```python
# def test_make_prediction(sample_input_data):
#     # Given
#     expected_first_prediction_value = 113422
#     expected_no_predictions = 1449
# 
#     # When
#     result = make_prediction(input_data=sample_input_data)
# 
#     # Then
#     predictions = result.get("predictions")
#     assert isinstance(predictions, list)
#     assert isinstance(predictions[0], np.float64)
#     assert result.get("errors") is None
#     assert len(predictions) == expected_no_predictions
#     assert math.isclose(predictions[0], expected_first_prediction_value, abs_tol=100)
# ```

# ## Versioning and packaging

# For packaging we have to look at a couple of files. You should not expect to write these files from scratch. Usually, these are automatically generated, or copied from projects you trust. First of these is [`pyproject.toml`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/pyproject.toml). Here we specify our build system, as well as settings for `pytest`, `black`, and `isort`. 
# 
# Next up is [`setup.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/setup.py). For this module we only usually just touch the package metadata. The file has some helpful comments on how to modify it. This module automatically sets the correct version from the `VERSION` file.
# 
# ```{margin}
# [`setup.py`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/setup.py)
# ```
# 
# ```python
# # Package meta-data.
# NAME = 'regression-model'
# DESCRIPTION = "Example regression model package for house prices."
# URL = "https://github.com/particle1331/model-deployment"
# EMAIL = "particle1331@gmail.com"
# AUTHOR = "particle1331"
# REQUIRES_PYTHON = ">=3.6.0"
# ```
# 
# Finally, we have [`MANIFEST.in`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/MANIFEST.in) which specifies which files to include and which files to exclude when building the package. The syntax should give you a general idea of what's happening.    

# 
# ```{margin}
# [`MANIFEST.in`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/MANIFEST.in)
# ```

# ```python
# include *.txt
# include *.md
# include *.pkl
# recursive-include ./regression_model/*
# 
# include regression_model/datasets/train.csv
# include regression_model/datasets/test.csv
# include regression_model/trained_models/*.pkl
# include regression_model/VERSION
# include regression_model/config.yml
# 
# include ./requirements/requirements.txt
# include ./requirements/test_requirements.txt
# exclude *.log
# exclude *.cfg
# 
# recursive-exclude * __pycache__
# recursive-exclude * *.py[co]
# ```

# Now, let us try building the package. 
# 
# ```
# $ python3 -m pip install --upgrade build
# $ python3 -m build
# ```

# This command should output a lot of text and once completed should generate two files in the `dist/` directory: a build distribution `.whl` file, and a source archive `tar.gz` for legacy builds. You should also see a `regression_model.egg-info/` directory which means the package has been successfully built. 

# ## Tooling

# In addition to `pytest`, we have the following tooling: `black` for opinionated styling, `flake8` for linting, `mypy` for type-checking, and `isort` for sorting imports. These are tools you should be familiar with by now by using `tox`. Styling makes the code easy to read, which links to maintainability. Automatic type checking reduces the possibility of bugs and mistakes that is more likely with a dynamically typed language such as Python.
# 
# 
# The settings for `mypy` is straightforward and can be found in [`mypy.ini`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/mypy.ini). Settings for `flake8` can be found in [`tox.ini`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/tox.ini). Finally, the settings for `pytest`, `black`, and `isort` can be found in [`pyproject.toml`](https://github.com/particle1331/model-deployment/blob/main/packages/regression_model/pyproject.toml). This concludes the discussion on production code. On the next notebook, we will look at a FastAPI application that consumes this package as a dependency.

# In[ ]:




