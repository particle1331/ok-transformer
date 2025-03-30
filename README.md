# OK TRANSFORMER

![build-status](https://img.shields.io/github/actions/workflow/status/particle1331/ok-transformer/publish-book.yml?branch=master)
![last-commit](https://img.shields.io/github/last-commit/particle1331/ok-transformer/master)
![python](https://shields.io/badge/python-3.12%20-blue) 
[![jupyter-book](https://raw.githubusercontent.com/jupyter-book/jupyter-book/refs/heads/main/docs/images/badge.svg)](https://jupyterbook.org/en/stable/intro.html)
&nbsp; ⭐ [![stars](https://img.shields.io/github/stars/particle1331/ok-transformer?style=social)](https://github.com/particle1331/ok-transformer) 

[OK TRANSFORMER](https://en.wikipedia.org/wiki/OK_Computer#Title) is a repository of Jupyter notebooks on **machine learning** **engineering** and **operations**. The notebooks contain some theory, end-to-end experiments, tests and benchmarks, as well as explorations of tools and frameworks in the larger ML ecosystem.
My goal in writing is to clarify my understanding[^1] and explore details[^2] that I might otherwise overlook. More pragmatically, the notebooks document patterns that worked, showing actual reproducible results.


[^1]: http://www.paulgraham.com/words.html
[^2]: http://www.paulgraham.com/getideas.html

## Making a local build

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv (optional)
git clone git@github.com:particle1331/ok-transformer.git && cd ok-transformer
make build
```
**Note:** The project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for environment and dependency management.

## Running the notebooks

The notebooks are located in `/docs/*`. 
Run them in a virtual environment created using:

```bash
uv venv --python 3.12
uv sync # install requirements
```

Use the resulting `.venv` as the Jupyter kernel. See [`pyproject.toml`](https://github.com/particle1331/ok-transformer/blob/master/pyproject.toml) for the dependency versions.

⚙️ The notebooks generally run end-to-end with reproducible results between runs. 
Exact output values may change due to external dependencies such as differences 
in hardware and dataset versions, or implementation quirks like [non-determinism](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility), but the conclusions should still hold.


## Hardware

Running the notebooks require modest hardware:

| **Component**       | **Kaggle Notebook**              | **MacBook Air M1**                  |
|---------------------|----------------------------------|-------------------------------------|
| **GPU 0**           | Tesla P100-PCIE-16GB             | Apple M1 Integrated GPU             |
| **CPU**             | Intel Xeon CPU @ 2.00GHz         | Apple M1 8-core CPU                 |
| **Core**            | 1                                | 4 high-performance, 4 efficiency    |
| **Threads per core**| 2                                | 1                                   |
| **L3 Cache**        | 38.5 MiB                         | 12 MiB                              |
| **Memory**          | 15 GB                            | 8 GB Unified Memory                 |

---
