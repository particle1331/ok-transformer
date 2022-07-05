
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
# Here e.g. https://www.kaggle.com/datasets/{USER}/{DATASET}
USER=
DATASET=
DATA_DIR=
mkdir ${DATA_DIR}
kaggle competitions download -c ${COMPETITION} -p ${DATA_DIR}
unzip ${DATA_DIR}/${USER}-${DATASET}.zip -d ${DATA_DIR}/${USER}-${DATASET} > /dev/null
rm ${DATA_DIR}/${USER}-${DATASET}.zip
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
