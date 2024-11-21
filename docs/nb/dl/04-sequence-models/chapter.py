import collections

class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.itot = ["<unk>"] + list(sorted(set(
            reserved_tokens +   # i.e. not subject to min_freq
            [token for token, freq in self.token_freqs if freq >= min_freq]
        )))
        self.ttoi = {tok: idx for idx, tok in enumerate(self.itot)}

    def __len__(self):
        return len(self.itot)
    
    def __getitem__(self, tokens: list[str]) -> list[int]:
        if isinstance(tokens, (list, tuple)):
            return [self.__getitem__(tok) for tok in tokens]
        else:
            return self.ttoi.get(tokens, self.unk)
            
    def to_tokens(self, indices) -> list[str]:
        if hasattr(indices, "__len__"):
            return [self.itot[int(index)] for index in indices]
        else:
            return self.itot[indices]

    @property
    def unk(self) -> int:
        return self.ttoi["<unk>"]

import re
import os
import requests


class TimeMachine:
    def __init__(self, download=False, token_level="char"):
        self.token_level = token_level
        self.filepath = "./data/time_machine.txt"
        if download or not os.path.exists(self.filepath):
            self._download()
        
    def _download(self):
        url = "https://www.gutenberg.org/cache/epub/35/pg35.txt"
        print(f"Downloading text from {url} ...", end=" ")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print("OK!")
        with open(self.filepath, "wb") as output:
            output.write(response.content)
        
    def _preprocess(self, text: str):
        s = "*** START OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***"
        e = "*** END OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***"
        text = text[text.find(s) + len(s): text.find(e)]
        text = re.sub('[^A-Za-z]+', ' ', text).lower().strip()
        return text
    
    def tokenize(self, text: str):
        return list(text) if self.token_level == "char" else text.split()
        
    def build(self, vocab=None):
        with open(self.filepath, "r") as f:
            raw_text = f.read()
        
        self.text = self._preprocess(raw_text)
        self.tokens = self.tokenize(self.text) 
        
        vocab = Vocab(self.tokens) if vocab is None else vocab
        corpus = vocab[self.tokens]
        return corpus, vocab

import re
import os
import requests

ENCODING = "utf-8-sig"


class ProjectGutenberg:
    def __init__(self, url: str, data_dir: str, download=False, token_level="char"):
        self.token_level = token_level
        self.filepath = f"{data_dir}/{url.split('/')[-1]}"
        if download or not os.path.exists(self.filepath):
            self.download(url, self.filepath)

    @staticmethod
    def get_title(filepath):
        with open(filepath, "r", encoding=ENCODING) as f:
            line = f.readline()
        prefix = "The Project Gutenberg eBook of"
        return line.replace(prefix, "").strip()

    @staticmethod
    def download(url, filepath):
        print(f"Downloading text from {url} ...", end=" ")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print("OK!")
        with open(filepath, "wb") as output:
            output.write(response.content)
        
    @staticmethod
    def preprocess(text: str, title: str):
        s = f"*** START OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***"
        e = f"*** END OF THE PROJECT GUTENBERG EBOOK {title.upper()} ***"
        text = text[text.find(s) + len(s): text.find(e)]
        text = re.sub('[^A-Za-z]+', ' ', text).lower().strip()
        return text
    
    @staticmethod
    def tokenize(text: str, token_level="char"):
        return list(text) if token_level == "char" else text.split()

    def build(self, vocab=None):
        with open(self.filepath, "r", encoding=ENCODING) as f:
            raw_text = f.read()
        
        self.title = self.get_title(self.filepath)
        self.text = self.preprocess(raw_text, self.title)
        self.tokens = self.tokenize(self.text, self.token_level)

        vocab = Vocab(self.tokens) if vocab is None else vocab
        corpus = vocab[self.tokens]
        return corpus, vocab

import requests
from bs4 import BeautifulSoup

def get_top100_books():
    url = "https://www.gutenberg.org/browse/scores/top#books-last30"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # scraping Top 100 EBooks yesterday section
    header = soup.find(id="books-last1")
    booklist = header.find_next("ol").find_all("a", href=True)

    # build url for plain text files
    text_urls = []
    base_url = "https://www.gutenberg.org"
    for link in booklist:
        idx = link["href"].split("/")[-1]
        text_urls.append(f"{base_url}/cache/epub/{idx}/pg{idx}.txt")

    return text_urls

import math
import torch
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib_inline import backend_inline

FRAC_LIMIT = 0.3
PAD_TOKEN = "."

DATASET_DIR = Path("./data").absolute()
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

warnings.simplefilter(action="ignore")
backend_inline.set_matplotlib_formats("svg")

def load_surnames(frac: float = FRAC_LIMIT, min_len=2) -> list[str]:
    """Load shuffled surnames from files into a list."""

    col = ["surname", "frequency_first", "frequency_second", "frequency_both"]
    filepaths = ["surnames_freq_ge_100.csv", "surnames_freq_ge_20_le_99.csv"]
    dfs = [pd.read_csv(DATASET_DIR / f, names=col, header=0) for f in filepaths]
    df = pd.concat(dfs, axis=0)[["surname"]].sample(frac=frac)
    df = df.reset_index(drop=True)
    df["surname"] = df["surname"].map(lambda s: s.lower())
    df["surname"] = df["surname"].map(lambda s: s.replace("de la", "dela"))
    df["surname"] = df["surname"].map(lambda s: s.replace(" ", "_"))
    df = df[["surname"]].dropna().astype(str)

    names = [
        n for n in df.surname.tolist() 
        if ("'" not in n) and ('ç' not in n) and (len(n) >= min_len)
    ]
    
    return names

import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, block_size: int, vocab=None):
        self.block_size = block_size
        self.xs = []
        self.ys = []
        self.vocab = vocab

    def vocab_size(self):
        return len(self.vocab)
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        x = self.encode(self.xs[idx])
        y = self.encode(self.ys[idx])[0]
        return x, y
        
    def decode(self, x: torch.tensor) -> str:
        return "".join(self.vocab.to_tokens(x))

    def encode(self, word: str) -> torch.tensor:
        return torch.tensor(self.vocab[list(word)]).long()

    def build(self, names: list[str]):
        xs, ys = [], []
        for name in names:
            context = [PAD_TOKEN] * self.block_size
            for c in name + PAD_TOKEN:
                xs.append(context)
                ys.append(c)
                context = context[1:] + [c]

        if self.vocab is None:
            self.vocab = Vocab(tokens=list(PAD_TOKEN + "".join(names)))

        self.xs = xs
        self.ys = ys
        return self

class CountingModel:
    def __init__(self, block_size: int, vocab_size: int, alpha=0.01):
        """Model of observed n-grams to estimate next char proba."""
        self.P = None                    # cond. prob
        self.N = None                    # counts
        self.alpha = alpha               # laplace smoothing
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __call__(self, x: torch.tensor) -> torch.tensor:
        # [[i_11, ... i_B1], [i_12, ..., i_B2]] for block_size = 2. 
        return torch.tensor(self.P[tuple(x.T)])    

    def fit(self, dataset: CharDataset):
        v = self.vocab_size
        n = self.block_size + 1     # +1 for output dim
        self.N = torch.zeros([v] * n, dtype=torch.int32)  
        for x, y in dataset:
            self.N[tuple(x)][y] += 1

        a = self.alpha
        self.P = (self.N + a)/ (self.N + a).sum(dim=-1, keepdim=True)

    def evaluate(self, dataset: CharDataset):
        loss = 0.0
        for x, y in dataset:
            loss += -torch.log(self(x[None, :])[0, y]).item()
        return loss / len(dataset)

def generate_name(
    model, 
    dataset: CharDataset, 
    min_len=2,
    max_len=100, 
    g=None, seed=2718
):
    """Generate names from a Markov process with cond prob from model."""
    if g is None:
        g = torch.Generator().manual_seed(seed)
    
    context = PAD_TOKEN * dataset.block_size
    out = []
    while len(context) < max_len:
        x = dataset.encode(context)[None, :]
        p = model(x)[0]
        j = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        c = dataset.decode(j)
        if c == PAD_TOKEN:
            if len(out) >= min_len:
                break
            else:
                continue
        
        out.append(c)
        context = context[1:] + c
    
    return "".join(out)

import numpy as np
from tqdm.notebook import tqdm
from contextlib import contextmanager
from torch.utils.data import DataLoader

DEVICE = "mps"


@contextmanager
def eval_context(model):
    """Temporarily set to eval mode inside context."""
    is_train = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(is_train)


class Trainer:
    def __init__(self,
        model, optim, loss_fn, scheduler=None, callbacks=[],
        device=DEVICE, verbose=True
    ):
        self.model = model.to(device)
        self.optim = optim
        self.device = device
        self.loss_fn = loss_fn
        self.train_log = {"loss": [], "loss_avg": []}
        self.valid_log = {"loss": []}
        self.verbose = verbose
        self.scheduler = scheduler
        self.callbacks = callbacks
    
    def __call__(self, x):
        return self.model(x.to(self.device))

    def forward(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        return self.model(x), y

    def train_step(self, batch):
        preds, y = self.forward(batch)
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return {"loss": loss}

    @torch.inference_mode()
    def valid_step(self, batch):
        preds, y = self.forward(batch)
        loss = self.loss_fn(preds, y, reduction="sum")
        return {"loss": loss}
    
    def run(self, epochs, train_loader, valid_loader, window_size=None):
        for e in tqdm(range(epochs)):
            for batch in train_loader:
                # optim and lr step
                output = self.train_step(batch)
                if self.scheduler:
                    self.scheduler.step()

                # step callbacks
                for callback in self.callbacks:
                    callback()

                # logs @ train step
                steps_per_epoch = len(train_loader)
                w = int(0.05 * steps_per_epoch) if not window_size else window_size
                self.train_log["loss"].append(output["loss"].item())
                self.train_log["loss_avg"].append(np.mean(self.train_log["loss"][-w:]))

            # logs @ epoch
            output = self.evaluate(valid_loader)
            self.valid_log["loss"].append(output["loss"])
            if self.verbose:
                print(f"[Epoch: {e+1:>0{int(len(str(epochs)))}d}/{epochs}]    loss: {self.train_log['loss_avg'][-1]:.4f}    val_loss: {self.valid_log['loss'][-1]:.4f}")

    def evaluate(self, data_loader):
        with eval_context(self.model):
            valid_loss = 0.0
            for batch in data_loader:
                output = self.valid_step(batch)
                valid_loss += output["loss"].item()

        return {"loss": valid_loss / len(data_loader.dataset)}

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        with eval_context(self.model):
            return self(x)

import torch
import numpy as np
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, dim_inputs, dim_hidden):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_inputs = dim_inputs
        self.W = nn.Parameter(torch.randn(dim_hidden, dim_hidden) / np.sqrt(dim_hidden))
        self.U = nn.Parameter(torch.randn(dim_inputs, dim_hidden) / np.sqrt(dim_inputs))
        self.b = nn.Parameter(torch.zeros(dim_hidden))

    def forward(self, x, state=None):
        x = x.transpose(0, 1)  # (B, T, d) -> (T, B, d)
        T, B, d = x.shape
        assert d == self.dim_inputs
        if state is None:
            state = torch.zeros(B, self.dim_hidden, device=x.device)
        else:
            assert state.shape == (B, self.dim_hidden)

        outs = []
        for t in range(T):
            state = torch.tanh(x[t] @ self.U + state @ self.W + self.b)
            outs.append(state)

        outs = torch.stack(outs)
        outs = outs.transpose(0, 1)
        return outs, state

import torch
import numpy as np
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, dim_inputs, dim_hidden):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_inputs = dim_inputs
        self.W = nn.Parameter(torch.randn(dim_hidden, dim_hidden) / np.sqrt(dim_hidden))
        self.U = nn.Parameter(torch.randn(dim_inputs, dim_hidden) / np.sqrt(dim_inputs))
        self.b = nn.Parameter(torch.zeros(dim_hidden))

    def forward(self, x, state=None):
        x = x.transpose(0, 1)  # (B, T, d) -> (T, B, d)
        T, B, d = x.shape
        assert d == self.dim_inputs
        if state is None:
            state = torch.zeros(B, self.dim_hidden, device=x.device)
        else:
            assert state.shape == (B, self.dim_hidden)

        outs = []
        for t in range(T):
            state = torch.tanh(x[t] @ self.U + state @ self.W + self.b)
            outs.append(state)

        outs = torch.stack(outs)
        outs = outs.transpose(0, 1)
        return outs, state

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, corpus: list, seq_len: int, vocab_size: int):
        super().__init__()
        self.corpus = corpus
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __getitem__(self, i):
        c = torch.tensor(self.corpus[i: i + self.seq_len + 1])
        x, y = c[:-1], c[1:]
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        return x, y
    
    def __len__(self):
        return len(self.corpus) - self.seq_len

class RNNLanguageModel(nn.Module):
    """RNN based language model."""
    def __init__(self, dim_inputs, dim_hidden, vocab_size):
        super().__init__()
        self.rnn = SimpleRNN(dim_inputs=dim_inputs, dim_hidden=dim_hidden)
        self.out_layer = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, state=None):
        outs, _ = self.rnn(x, state)
        logits = self.out_layer(outs)   # (B, T, H) -> (B, T, C)
        return logits.permute(0, 2, 1)  # F.cross_entropy expects (B, C, T)

class RNNLanguageModel(nn.Module):
    """RNN based language model."""
    def __init__(self, dim_inputs, dim_hidden, vocab_size):
        super().__init__()
        self.rnn = SimpleRNN(dim_inputs=dim_inputs, dim_hidden=dim_hidden)
        self.ffn = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, state=None):
        outs, _ = self.rnn(x, state)
        logits = self.ffn(outs)         # (B, T, H) -> (B, T, C)
        return logits.permute(0, 2, 1)  # F.cross_entropy expects (B, C, T)

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, corpus: list, seq_len: int, vocab_size: int):
        super().__init__()
        self.corpus = corpus
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __getitem__(self, i):
        c = torch.tensor(self.corpus[i: i + self.seq_len + 1])
        x, y = c[:-1], c[1:]
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        return x, y
    
    def __len__(self):
        return len(self.corpus) - self.seq_len

class RNNLanguageModel(nn.Module):
    """RNN based language model."""
    def __init__(self, dim_inputs, dim_hidden, vocab_size):
        super().__init__()
        self.rnn = SimpleRNN(dim_inputs=dim_inputs, dim_hidden=dim_hidden)
        self.lin = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, state=None):
        outs, _ = self.rnn(x, state)
        logits = self.lin(outs)         # (B, T, H) -> (B, T, C)
        return logits.permute(0, 2, 1)  # F.cross_entropy expects (B, C, T)

class RNNLanguageModel(nn.Module):
    """RNN based language model."""
    def __init__(self, dim_inputs, dim_hidden, vocab_size):
        super().__init__()
        self.rnn = SimpleRNN(dim_inputs=dim_inputs, dim_hidden=dim_hidden)
        self.lin = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, state=None):
        outs, _ = self.rnn(x, state)
        logits = self.lin(outs)         # (B, T, H) -> (B, T, C)
        return logits.permute(0, 2, 1)  # F.cross_entropy expects (B, C, T)

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, corpus: list, seq_len: int, vocab_size: int):
        super().__init__()
        self.corpus = corpus
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __getitem__(self, i):
        c = torch.tensor(self.corpus[i: i + self.seq_len + 1])
        x, y = c[:-1], c[1:]
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        return x, y
    
    def __len__(self):
        return len(self.corpus) - self.seq_len

class RNNLanguageModel(nn.Module):
    """RNN based language model."""
    def __init__(self, dim_inputs, dim_hidden, vocab_size):
        super().__init__()
        self.rnn = SimpleRNN(dim_inputs=dim_inputs, dim_hidden=dim_hidden)
        self.linear = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, state=None):
        outs, _ = self.rnn(x, state)
        logits = self.linear(outs)         # (B, T, H) -> (B, T, C)
        return logits.permute(0, 2, 1)  # F.cross_entropy expects (B, C, T)

class RNNLanguageModel(nn.Module):
    """RNN based language model."""
    def __init__(self, dim_inputs, dim_hidden, vocab_size):
        super().__init__()
        self.rnn = SimpleRNN(dim_inputs=dim_inputs, dim_hidden=dim_hidden)
        self.linear = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, state=None):
        outs, _ = self.rnn(x, state)
        logits = self.linear(outs)         # (B, T, H) -> (B, T, C)
        return logits.permute(0, 2, 1)  # F.cross_entropy expects (B, C, T)

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, corpus: list, seq_len: int, vocab_size: int):
        super().__init__()
        self.corpus = corpus
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __getitem__(self, i):
        c = torch.tensor(self.corpus[i: i + self.seq_len + 1])
        x, y = c[:-1], c[1:]
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        return x, y
    
    def __len__(self):
        return len(self.corpus) - self.seq_len

import torch
import numpy as np
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, dim_inputs, dim_hidden):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_inputs = dim_inputs
        self.W = nn.Parameter(torch.randn(dim_hidden, dim_hidden) / np.sqrt(dim_hidden))
        self.U = nn.Parameter(torch.randn(dim_inputs, dim_hidden) / np.sqrt(dim_inputs))
        self.b = nn.Parameter(torch.zeros(dim_hidden))

    def forward(self, x, state=None):
        x = x.transpose(0, 1)  # (B, T, d) -> (T, B, d)
        T, B, d = x.shape
        assert d == self.dim_inputs
        if state is None:
            state = torch.zeros(B, self.dim_hidden, device=x.device)
        else:
            assert state.shape == (B, self.dim_hidden)

        outs = []
        for t in range(T):
            state = torch.tanh(x[t] @ self.U + state @ self.W + self.b)
            outs.append(state)

        outs = torch.stack(outs)
        outs = outs.transpose(0, 1)
        return outs, state

class RNNLanguageModel(nn.Module):
    """RNN based language model."""
    def __init__(self, dim_inputs, dim_hidden, vocab_size):
        super().__init__()
        self.rnn = SimpleRNN(dim_inputs, dim_hidden)
        self.linear = nn.Linear(dim_hidden, vocab_size)

    def forward(self, x, state=None):
        outs, _ = self.rnn(x, state)
        logits = self.linear(outs)         # (B, T, H) -> (B, T, C)
        return logits.permute(0, 2, 1)  # F.cross_entropy expects (B, C, T)

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, corpus: list, seq_len: int, vocab_size: int):
        super().__init__()
        self.corpus = corpus
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __getitem__(self, i):
        c = torch.tensor(self.corpus[i: i + self.seq_len + 1])
        x, y = c[:-1], c[1:]
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        return x, y
    
    def __len__(self):
        return len(self.corpus) - self.seq_len

