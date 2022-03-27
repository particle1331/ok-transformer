from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    DATASET_DIR = Path(__file__).parent.resolve() / "data"
    TRAINED_MODELS_DIR = Path(__file__).parent.resolve() / "trained_models"

    # Create directories
    TRAINED_MODELS_DIR.mkdir(exist_ok=True)
    DATASET_DIR.mkdir(exist_ok=True)

    def set_tensorflow_seeds(self, seed=0):
        import tensorflow as tf
        import numpy as np
        import random as python_random

        np.random.seed(seed)
        python_random.seed(seed)
        tf.random.set_seed(seed)

    def set_matplotlib(self):
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats('svg', 'pdf')

    def set_ignore_warnings(self):
        import warnings
        warnings.simplefilter(action='ignore')

    def list_tensorflow_devices(self):
        import tensorflow as tf
        return tf.config.list_physical_devices()


config = Config()