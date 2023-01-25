# OK Transformer

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fparticle1331%2Fok-transformer%2Fbadge%3Fref%3Dmaster&label=build&logo=none)](https://actions-badge.atrox.dev/particle1331/ok-transformer/goto?ref=master)
![Last Commit](https://img.shields.io/github/last-commit/particle1331/ok-transformer/master)
![python](https://img.shields.io/github/pipenv/locked/python-version/particle1331/ok-transformer)
![jupyter-book](https://github.com/executablebooks/jupyter-book/raw/master/docs/images/badge.svg)
[![Stars](https://img.shields.io/github/stars/particle1331/ok-transformer?style=social)](https://github.com/particle1331/ok-transformer) 

Entry point: [**OK Transformer** website](https://particle1331.github.io/ok-transformer/intro.html)

<br>

These are self-contained notebooks (code-wise) on topics in machine learning engineering such as neural networks, language modeling, TensorFlow/Keras and PyTorch deep learning, high-performance Python, data engineering and MLOps. The notebooks are ensured run end-to-end and have reproducible results before being published. This last condition takes a significant amount of effort on my part.

<br>

## Local build

The collection is built using [jupyter-book](https://github.com/executablebooks/jupyter-book) and hosted as a [Github Pages](https://jupyterbook.org/en/stable/publish/gh-pages.html) website. After installing `jupyter-book`, making a local build of the book is easy:

```
git clone git@github.com:particle1331/ok-transformer.git
cd ok-transformer
make docs
```

## Dependencies

Nothing too strict. But the result of doing `pip list` on my local conda env is:

```text
cmaes                         0.8.2
docker                        5.0.3
docker-compose                1.25.5
fastapi                       0.75.2
kaggle                        1.5.12
kaleido                       0.2.1
keras                         2.8.0
matplotlib                    3.5.1
mlflow                        1.26.1
numpy                         1.22.4
optuna                        2.10.0
pandas                        1.4.2
pipenv                        2022.6.7
prefect                       2.0b5
pydantic                      1.8.2
scikit-learn                  1.0.2
scipy                         1.8.1
seaborn                       0.11.2
SQLAlchemy                    1.4.37
statsmodels                   0.13.2
tensorboard                   2.8.0
tensorflow-datasets           4.5.2
tensorflow-macos              2.8.0
tensorflow-metadata           1.7.0
tensorflow-metal              0.4.0
tf-estimator-nightly          2.8.0.dev2021122109
torch                         1.13.0
torchvision                   0.14.0
uvicorn                       0.17.6
xgboost                       1.6.0.dev0
```

## Hardware

I generally just use a Macbook Air with an M1 chip, although sometimes I will rent an instance in [[λ] Lambda](cloud.lambdalabs.com) or use a [🇰 Kaggle kernel](https://www.kaggle.com/code) for tasks where it requires more compute to observe something. See [this](https://github.com/particle1331/M1-tensorflow-benchmark#mlp-benchmark) for relevant performance benchmarks with the M1. The specs of a P100 Kaggle kernel is shown below:

```
GPU 0: Tesla P100-PCIE-16GB (UUID: GPU-543c532b-c511-c675-a565-bf01208405e0)
Model name:                      Intel(R) Xeon(R) CPU @ 2.00GHz
Socket(s):                       1
Core(s) per socket:              1
Thread(s) per core:              2
L3 cache:                        38.5 MiB
CPU MHz:                         2000.188
MemAvailable:   15212104 kB
Avail
67G
```