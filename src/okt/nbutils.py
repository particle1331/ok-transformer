from pathlib import Path
from matplotlib_inline import backend_inline
import matplotlib.pyplot as plt


def figure(
    file: str, /,
    caption_lead="", 
    caption_body="",
    dir="img",
    width=100, 
    align="center"
):
    """Snippet to create a figure in markdown format."""
    filename = ".".join(file.split(".")[:-1])
    if caption_lead:
        fig_caption = f"**{caption_lead}** {caption_body}"
    else:
        fig_caption = caption_body

    assert Path(f"./{dir}/{file}").exists(), f"File {dir}/{file} does not exist."
    print(
f"""
:::{{figure}} ./{dir}/{file}
---
name: {filename}
width: {width}%
align: {align}
---
{fig_caption}
:::"""
    )


def savefig(file: str, /, dir="plots"):
    error = "File format not supported."
    assert file.split(".")[-1] in ["png", "pdf", "svg"], error
    print(
f"""
plt.savefig("./{dir}/{file}", bbox_inches="tight")
plt.close("all")
"""
    )


def init():
    print(r"""
%load_ext autoreload
%autoreload 2
from okt.nbutils import set_plot_params
from okt.nn.utils import get_device, set_seed
set_plot_params()

from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT_DIR = Path().resolve().parent.parent
LOCAL_DIR = ROOT_DIR / ".local"
DATASET_DIR = LOCAL_DIR / "data"
ARTIFACTS_DIR = LOCAL_DIR / "artifacts"
warnings.simplefilter(action="ignore")

RANDOM_SEED = 0
DEVICE = get_device()
print(f"Using device: {DEVICE}")
set_seed(RANDOM_SEED)
"""
)


def set_plot_params(
    rcparams={
        "font.size": 7,
        "font.family": "monospace",
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
    },
    plot_format = "retina"
):
    plt.rcParams.update(rcparams)
    backend_inline.set_matplotlib_formats(plot_format)
