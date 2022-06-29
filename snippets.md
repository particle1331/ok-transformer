
Ignoring warnings

```python
from pandas.core.common import SettingWithCopyWarning
from matplotlib_inline import backend_inline

import warnings

backend_inline.set_matplotlib_formats('svg')
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
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
kaggle datasets download -d ${USER}/${DATASET} -p ${DATA_PATH}
unzip ${DATA_PATH}.zip -d ${DATA_PATH} > /dev/null
rm ${DATA_PATH}.zip
```

Download from Kaggle competition
```bash
# Here e.g. https://www.kaggle.com/c/{COMPETITION}
COMPETITION=
kaggle competitions download -c ${COMPETITION} -p ${DATA_PATH}
unzip ${DATA_PATH}.zip -d ${DATA_PATH} > /dev/null
rm ${DATA_PATH}.zip
```
