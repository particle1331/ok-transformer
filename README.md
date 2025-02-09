# OK Transformer

[![build-status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fparticle1331%2Fok-transformer%2Fbadge%3Fref%3Dmaster&label=build&logo=none)](https://actions-badge.atrox.dev/particle1331/ok-transformer/goto?ref=master)
![last-commit](https://img.shields.io/github/last-commit/particle1331/ok-transformer/master)
![python](https://shields.io/badge/python-3.12%20-blue) 
[![jupyter-book](https://raw.githubusercontent.com/jupyter-book/jupyter-book/refs/heads/main/docs/images/badge.svg)](https://jupyterbook.org/en/stable/intro.html)
[![stars](https://img.shields.io/github/stars/particle1331/ok-transformer?style=social)](https://github.com/particle1331/ok-transformer) 

Entry point: [<ins>**OK Transformer** website</ins>](https://particle1331.github.io/ok-transformer/intro.html)

A collection of self-contained notebooks on machine learning theory, engineering, and operations. I cover topics that frequently come up as building blocks for applications or further theory. I also explore areas where I seek to [clarify my understanding](http://www.paulgraham.com/words.html) or [delve into details](http://www.paulgraham.com/getideas.html) that I find particularly interesting or intriguing.


## Making a local build

The book can be built locally using [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```
git clone git@github.com:particle1331/ok-transformer.git && cd ok-transformer
make build
```

## Running the notebooks

The notebooks are located in `/docs/nb`. 
Create a virtual environment using:

```
uv venv
uv sync
```

Use the resulting `.venv` as Jupyter kernel. See [`pyproject.toml`](https://github.com/particle1331/ok-transformer/blob/master/pyproject.toml) for the library versions installed.

⚙️ The notebooks generally run end-to-end with reproducible results between runs. 
Exact output values may change due to external dependencies such as differences 
with hardware and dataset versions, or implementation quirks like [non-determinism](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility), but the conclusions should still hold.

## Hardware

The hardware requirements used to run the notebooks are modest:

| **Component**       | **Kaggle Notebook**   | **MacBook Air M1**                  |
|---------------------|----------------------------------|-------------------------------------|
| **GPU 0**          | Tesla P100-PCIE-16GB            | Apple M1 Integrated GPU |
| **CPU**            | Intel Xeon CPU @ 2.00GHz        | Apple M1 8-core CPU                |
| **Core**           | 1                                | 4 high-performance, 4 efficiency |
| **Threads per core**| 2                                | 1                                   |
| **L3 Cache**       | 38.5 MiB                        | 12 MiB                             |
| **Memory**         | 15 GB                           | 8 GB Unified Memory             |
