
Format matplotlib plots
```python
from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')
```

Ignoring warnings

```python
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="once", category=SettingWithCopyWarning)
```

Set random seeds

```python
import numpy as np
import tensorflow as tf
import random

seed = 42
np.random.seed(seed)
random.random.seed(seed)
tf.random.set_seed(seed)
```

List TF devices

```python
import tensorflow as tf
tf.config.list_physical_devices()
```

Creating paths

```python
import pathlib 

WORKDIR = pathlib.Path().absolute()
DATA_PATH = WORKDIR / "data"
MODELS_PATH = WORKDIR / "models"

# Create directories
MODELS_PATH.mkdir(exist_ok=True)
DATA_PATH.mkdir(exist_ok=True)
```


Download from Kaggle datasets
```bash 
# Here e.g. https://www.kaggle.com/datasets/waifuai/cat2dog
USER="waifuai"
DATASET="cat2dog"
DATA_DIR=./data
mkdir ${DATA_DIR}
kaggle datasets download -d ${USER}/${DATASET} -p ${DATA_DIR}
unzip ${DATA_DIR}/${DATASET}.zip -d ${DATA_DIR}/${DATASET} > /dev/null
rm ${DATA_DIR}/${DATASET}.zip
```

Download from Kaggle competition
```bash
COMPETITION=house-prices-advanced-regression-techniques
DATA_DIR=./data
mkdir ${DATA_DIR}
kaggle competitions download -c ${COMPETITION} -p ${DATA_DIR}
unzip ${DATA_DIR}/${COMPETITION}.zip -d ${DATA_DIR}/${COMPETITION} > /dev/null
rm ${DATA_DIR}/${COMPETITION}.zip
```

Header for each article:
````
![Status](https://img.shields.io/static/v1.svg?label=Status&message=Ongoing&color=orange)
[![Source](https://img.shields.io/static/v1.svg?label=GitHub&message=Source&color=181717&logo=GitHub)](https://github.com/particle1331/inefficient-networks/blob/master/docs/notebooks/mlops/04-deployment)
[![Stars](https://img.shields.io/github/stars/particle1331/inefficient-networks?style=social)](https://github.com/particle1331/inefficient-networks)

```text
𝗔𝘁𝘁𝗿𝗶𝗯𝘂𝘁𝗶𝗼𝗻: Notes for Module 6 of the MLOps Zoomcamp (2022) by DataTalks.Club.
```

---

## Introduction
````


Adding scripts with margin links.
````
```{margin}
[`predict.py`](https://github.com/particle1331/inefficient-networks/blob/383314b4c5e01fe9cc9d65b9ce1b9b90abb04001/docs/notebooks/mlops/04-deployment/ride_duration/predict.py#L10-L16)
```
```python
def load_model(experiment_id, run_id):
    """Get model from our S3 artifacts store."""

    source = f"s3://mlflow-models-ron/{experiment_id}/{run_id}/artifacts/model"
    model = mlflow.pyfunc.load_model(source)

    return model
```
````


Adding figures
````
```{margin}
[Source](sourceurl)
```
```{figure} ../../../img/pypi.png
---
width: 40em
---
Our model package in the Python package index. 🐍
```
````

Building and running Docker on M1 MacOS:
```
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```
