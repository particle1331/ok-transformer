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
[![Source](https://img.shields.io/static/v1.svg?label=GitHub&message=Source&color=181717&logo=GitHub)](https://github.com/particle1331/ok-transformer/blob/master/docs/nb/mlops/04-deployment)
[![Stars](https://img.shields.io/github/stars/particle1331/ok-transformer?style=social)](https://github.com/particle1331/ok-transformer)

```text
𝗔𝘁𝘁𝗿𝗶𝗯𝘂𝘁𝗶𝗼𝗻: Notes for Module 6 of the MLOps Zoomcamp (2022) by DataTalks.Club.
```

---

## Introduction
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


Add path to end of `.zshrc`
```
echo 'export PATH="/opt/homebrew/opt/redis@6.2/bin:$PATH"' >> ~/.zshrc
```

Notebook diff with commits
```
nbdiff-web 58689b5d e6b603d6 docs/nb/fundamentals/backpropagation.ipynb
```

Comparing commits on github
```
https://github.com/particle1331/ok-transformer/compare/58689b5d..e6b603d6
```


Kaggle kernel hardware
```
!nvidia-smi -L 
!lscpu | grep 'Model name'
!lscpu | grep 'Socket(s):'
!lscpu | grep 'Core(s) per socket'
!lscpu | grep 'Thread(s) per core'
!lscpu | grep 'L3 cache'
!lscpu | grep MHz
!cat /proc/meminfo | grep 'MemAvailable'
!df -h / | awk '{print $4}'
```


- private feature branches: git rebase / git pull --rebase
- public branches (e.g. main): git merge

tooling
- fzf
- zsh
- iterm
- tmux


Properties setters and getters
```python
class Test:
    def __init__(self):
        self._value = 100

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, x):
        if x < 0:
            raise ValueError("value cannot be negative.")
        self._value = x
```
```python
t = Test()
t.value = -3 # exception
```

Processes from terminal:
```
$ top
```

Pytest inside notebooks:
```python
import ipytest
ipytest.autoconfig()

def test_addition():
    assert 1 + 1 == 2

assert ipytest.run("-vv") == 0
```

Colima as docker desktop replacement for macos
```
colima start --cpu 4 --memory 4
colima stop

# using profiles
colima -p new-profile start
colima list
```

Custom jupyter extensions (e.g. `%%save`)

```python
# /Users/particle1331/.ipython/profile_default/startup/my_extension.py
from IPython.core.magic import register_cell_magic

@register_cell_magic
def save(line, cell):
    # Append the cell content to the file
    with open("chapter.py", 'a') as f:
        f.write(cell + '\n')

    # Execute the cell content
    exec(cell, globals())

    return Code(cell, language="python")
```

```python
# /Users/particle1331/.ipython/profile_default/startup/load_my_extension.py
from IPython import get_ipython

get_ipython().run_line_magic('load_ext', 'my_extension')
```

```
%matplotlib inline
%config InlineBackend.figure_format = 'svg'
```

```
ruff-check:
	- pdm run ruff check $(path)
	- pdm run ruff check --select I --diff $(path)
	- pdm run ruff format --diff $(path)

ruff-format:
	pdm run ruff check --select I --fix $(path)
	pdm run ruff format $(path)
```

Plotting a Go board:
```python
import matplotlib.pyplot as plt
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

# create a 8" x 8" board
fig = plt.figure(figsize=[8,8])
fig.patch.set_facecolor((1,1,.8))

ax = fig.add_subplot(111)

# draw the grid
for x in range(19):
    ax.plot([x, x], [0,18], 'k')
for y in range(19):
    ax.plot([0, 18], [y,y], 'k')

# scale the axis area to fill the whole figure
ax.set_position([0,0,1,1])

# get rid of axes and everything (the figure background will show through)
ax.set_axis_off()

# scale the plot area conveniently (the board is in 0,0..18,18)
ax.set_xlim(-1,19)
ax.set_ylim(-1,19)

# draw Go stones at (10,10) and (13,16)
s1, = ax.plot(10,10,'o',markersize=30, markeredgecolor=(0,0,0), markerfacecolor='w', markeredgewidth=2)
s2, = ax.plot(13,16,'o',markersize=30, markeredgecolor=(.5,.5,.5), markerfacecolor='k', markeredgewidth=2)
```
