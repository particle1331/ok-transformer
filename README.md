# OK Transformer

[![build-status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fparticle1331%2Fok-transformer%2Fbadge%3Fref%3Dmaster&label=build&logo=none)](https://actions-badge.atrox.dev/particle1331/ok-transformer/goto?ref=master)
![last-commit](https://img.shields.io/github/last-commit/particle1331/ok-transformer/master)
![python](https://shields.io/badge/python-3.10%20-blue)
[![stars](https://img.shields.io/github/stars/particle1331/ok-transformer?style=social)](https://github.com/particle1331/ok-transformer) 

[Entry point](https://particle1331.github.io/ok-transformer/intro.html)

A collection of notes, code, and experiments on understanding ML engineering and operations.

<br>


## Making a local build

```
git clone git@github.com:particle1331/ok-transformer.git && cd ok-transformer
pip install -U tox
tox -e build
```

## Running the notebooks

The notebooks can be found in `docs/nb`. 
To run them, create a venv using [`pdm`](https://github.com/pdm-project/pdm):

```
pip install -U pdm
pdm install
```

Use the resulting `.venv` as Jupyter kernel. 
The following libraries will be installed (see `pdm.lock`):

```text
╭────────────────────────────────────┬────────────────╮
│ fastapi                            │ 0.111.0        │
│ Flask                              │ 3.0.3          │
│ keras                              │ 2.15.0         │
│ lightning                          │ 2.3.0          │
│ matplotlib                         │ 3.9.0          │
│ mlflow                             │ 2.13.2         │
│ numpy                              │ 1.26.4         │
│ optuna                             │ 3.6.1          │
│ pandas                             │ 2.2.2          │
│ scikit-learn                       │ 1.5.0          │
│ scipy                              │ 1.13.1         │
│ seaborn                            │ 0.13.2         │
│ tensorflow                         │ 2.15.1         │
│ tensorflow-datasets                │ 4.9.6          │
│ tensorflow-estimator               │ 2.15.0         │
│ torch                              │ 2.2.2          │
│ torchaudio                         │ 2.2.2          │
│ torchinfo                          │ 1.8.0          │
│ torchmetrics                       │ 1.4.0.post0    │
│ torchvision                        │ 0.17.2         │
│ uvicorn                            │ 0.30.1         │
│ xgboost                            │ 2.0.3          │
╰────────────────────────────────────┴────────────────╯
```

## Hardware

```
GPU 0:                           Tesla P100-PCIE-16GB
CPU:                             Intel(R) Xeon(R) CPU @ 2.00GHz
Core:                            1
Threads per core:                2
L3 cache:                        38.5 MiB
Memory:                          15 Gb
```
