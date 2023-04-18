# OK Transformer

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fparticle1331%2Fok-transformer%2Fbadge%3Fref%3Dmaster&label=build&logo=none)](https://actions-badge.atrox.dev/particle1331/ok-transformer/goto?ref=master)
![Last Commit](https://img.shields.io/github/last-commit/particle1331/ok-transformer/master)
![python](https://img.shields.io/github/pipenv/locked/python-version/particle1331/ok-transformer)
![jupyter-book](https://github.com/executablebooks/jupyter-book/raw/master/docs/images/badge.svg)
[![Stars](https://img.shields.io/github/stars/particle1331/ok-transformer?style=social)](https://github.com/particle1331/ok-transformer) 

Entry point: [**OK Transformer** website](https://particle1331.github.io/ok-transformer/intro.html)

<br>

A collection of self-contained notebooks on topics related to machine learning engineering and operations. I take great pains (or many re-runs) to ensure 
that the notebooks run end-to-end with reasonably reproducible results across runs. Of course, there are a lot of moving parts so things might change, e.g. with different hardware, but correct conclusions should still hold (even if the values are different). 

Writing takes a nontrivial amount of time and effort, and so there needs to be some 
justification. The simple answer is that I find I learn better by writing. 
After all, only about a small fraction of things that you consider to convince
yourself of something distills into writing. I also get to [focus](https://en.wikipedia.org/wiki/Attention_economy). 
I often find my ideas to be wrong only in the process of writing about it
or justifying it with computation.
This process leads to more ideas which tends to be either [more correct](http://www.paulgraham.com/words.html) or [more concise](http://www.paulgraham.com/writing44.html) (which is another sign of correctness). 
This is also an attempt to learn in public, which is a scary thing when you think
about it.

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

Should be okay to within minor releases:

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

P100 Kaggle kernel or equivalent with the ff. specs should be sufficient:

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
