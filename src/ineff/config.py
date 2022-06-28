from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    PACKAGE_ROOT = Path(__file__).parent.resolve()
    DATA_PATH = PACKAGE_ROOT / "data"
    MODELS_PATH = PACKAGE_ROOT / "models"

    # Create directories
    MODELS_PATH.mkdir(exist_ok=True)
    DATA_PATH.mkdir(exist_ok=True)


    def set_tensorflow_seeds(self, seed=0):
        import tensorflow as tf
        import numpy as np
        import random as python_random

        np.random.seed(seed)
        python_random.seed(seed)
        tf.random.set_seed(seed)

    def set_matplotlib(self, format="svg"):
        from matplotlib_inline import backend_inline
        backend_inline.set_matplotlib_formats(format)

    def set_ignore_warnings(self, action='once'):
        import warnings
        warnings.simplefilter(action=action)

    def list_tensorflow_devices(self):
        import tensorflow as tf
        return tf.config.list_physical_devices()


config = Config()