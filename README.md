# OK Transformer

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fparticle1331%2Fok-transformer%2Fbadge%3Fref%3Dmaster&label=build&logo=none)](https://actions-badge.atrox.dev/particle1331/ok-transformer/goto?ref=master)
![Last Commit](https://img.shields.io/github/last-commit/particle1331/ok-transformer/master)
![python](https://img.shields.io/github/pipenv/locked/python-version/particle1331/ok-transformer)
![jupyter-book](https://github.com/executablebooks/jupyter-book/raw/master/docs/images/badge.svg)
[![Stars](https://img.shields.io/github/stars/particle1331/ok-transformer?style=social)](https://github.com/particle1331/ok-transformer) 

Entry point: [**OK Transformer** website](https://particle1331.github.io/ok-transformer/intro.html)

<br>

A collection of self-contained notebooks on topics in machine learning engineering. This includes neural networks and language modeling, deep learning in TensorFlow/Keras and PyTorch, high-performance Python, model deployment and MLOps. The notebooks run end-to-end with generally reproducible results across runs.

<br>

## Local build

Making a local build:

```
git clone git@github.com:particle1331/ok-transformer.git
cd ok-transformer
pip install -r build-requirements.txt
make docs
```

## Dependencies

Should be okay to within micro (perhaps even minor) releases:

```text
docker                        5.0.3
docker-compose                1.25.5
fastapi                       0.75.2
keras                         2.8.0
matplotlib                    3.5.1
mlflow                        1.26.1
numpy                         1.22.4
optuna                        2.10.0
pandas                        1.4.2
pipenv                        2022.6.7
prefect                       2.0b5
scikit-learn                  1.0.2
seaborn                       0.11.2
tensorflow-datasets           4.5.2
tensorflow-macos              2.8.0
tensorflow-metal              0.4.0
torch                         1.13.0
torchvision                   0.14.0
uvicorn                       0.17.6
xgboost                       1.6.0.dev0
```

## Hardware

I use a Macbook Air with an M1 chip for prototyping. For more compute, I rent an instance in [[λ] Lambda](cloud.lambdalabs.com) or use a [🇰 Kaggle kernel](https://www.kaggle.com/code) (e.g. to observe something not observable using a small model or small dataset). See [this](https://github.com/particle1331/M1-tensorflow-benchmark#mlp-benchmark) for relevant performance benchmarks with the M1. The specs of a P100 Kaggle kernel is as follows:

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
