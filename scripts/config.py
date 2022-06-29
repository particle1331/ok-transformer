from pydantic import BaseModel
from pathlib import Path


from pandas.core.common import SettingWithCopyWarning
from matplotlib_inline import backend_inline

import warnings

backend_inline.set_matplotlib_formats('svg')
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



def create_paths():

    import pathlib 

    WORKDIR = pathlib.Path().absolute()
    DATA_PATH = WORKDIR / "data"
    MODELS_PATH = WORKDIR / "models"

    # Create directories
    MODELS_PATH.mkdir(exist_ok=True)
    DATA_PATH.mkdir(exist_ok=True)


def set_seeds(self, seed=0):
        
    import tensorflow as tf
    import numpy as np
    import random

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


class Config(BaseModel):
    

    

    def set_matplotlib_format(self, format="svg"):

    def set_ignore_warnings(self, action='once'):

    def list_tensorflow_devices(self):
        import tensorflow as tf
        return tf.config.list_physical_devices()


config = Config()
