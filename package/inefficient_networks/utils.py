import warnings
import pathlib
import os

from inefficient_networks.config import config


def set_tf_seeds(seed=0):
    import tensorflow as tf
    import numpy as np
    import random as python_random

    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def set_matplotlib():
    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('svg', 'pdf') # <!>


def set_ignore_warnings():
    warnings.simplefilter(action='ignore')


def list_tf_devices():
    import tensorflow as tf
    return tf.config.list_physical_devices()


def download_kaggle_dataset(*, dataset: str):
    """Here dataset is the portion of the URL https://www.kaggle.com/datasets/{dataset}."""
    os.system(f"kaggle datasets download -d {dataset} -p {config.DATASET_DIR}")


def download_kaggle_competition(*, competition: str):
    """Here dataset is the portion of the URL https://www.kaggle.com/c/{competition}."""
    os.system(f"kaggle competitions download -c {competition} -p {config.DATASET_DIR}")
