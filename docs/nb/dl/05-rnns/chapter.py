import torch
import torch.nn as nn
import numpy as np
import random

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MPS = torch.backends.mps.is_available()
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0") if CUDA else torch.device("mps") if MPS else torch.device("cpu")

class RNNBase(nn.Module):
    """Base class for recurrent units, e.g. RNN, LSTM, GRU, etc."""
    def __init__(self, inputs_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inputs_dim = inputs_dim
        
    def init_state(self, x):
        raise NotImplementedError
    
    def compute(self, x, state):
        raise NotImplementedError

    def forward(self, x, state=None):
        state = self.init_state(x) if state is None else state
        outs, state = self.compute(x, state)
        return outs, state

class RNN(RNNBase):
    """Simple RNN unit."""
    def __init__(self, inputs_dim: int, hidden_dim: int):
        super().__init__(inputs_dim, hidden_dim)
        self.U = nn.Parameter(torch.randn(inputs_dim, hidden_dim) / np.sqrt(inputs_dim))
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def init_state(self, x):
        B = x.shape[1]
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        return h
    
    def compute(self, x, state):
        h = state
        T = x.shape[0]
        outs = []
        for t in range(T):
            h = torch.tanh(x[t] @ self.U + h @ self.W + self.b)
            outs.append(h)
        return torch.stack(outs), h

import torch
import torch.nn as nn
from typing import Type
from functools import partial


class RNNLanguageModel(nn.Module):
    def __init__(self, 
        cell: Type[RNNBase],
        inputs_dim: int,
        hidden_dim: int,
        vocab_size: int,
        **kwargs
    ):
        super().__init__()
        self.cell = cell(inputs_dim, hidden_dim, **kwargs)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, state=None, return_state=False):
        outs, state = self.cell(x, state)
        outs = self.linear(outs)    # (T, B, H) -> (T, B, C)
        return outs if not return_state else (outs, state)


LanguageModel = lambda cell: partial(RNNLanguageModel, cell)

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

import re
import os
import requests
import collections
from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


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


class TimeMachine:
    def __init__(self, download=False, path=None, token_level="char"):
        DEFAULT_PATH = str((DATA_DIR / "time_machine.txt").absolute())
        self.token_level = token_level
        self.filepath = path or DEFAULT_PATH
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

def collate_fn(batch):
    """Transforming the data to sequence-first format."""
    x, y = zip(*batch)
    x = torch.stack(x, 1)      # (T, B, vocab_size)
    y = torch.stack(y, 1)      # (T, B)
    return x, y

import torch.nn.functional as F

def clip_grad_norm(model, max_norm: float):
    """Calculate norm on concatenated params. Modify params in-place."""
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > max_norm:
        for p in params:
            p.grad[:] *= max_norm / norm   # [:] = shallow copy, in-place

def train_step(model, optim, x, y, max_norm) -> float:
    target = y.transpose(0, 1)
    output = model(x).permute(1, 2, 0)
    loss = F.cross_entropy(output, target)
    loss.backward()
    
    clip_grad_norm(model, max_norm=max_norm)
    optim.step()
    optim.zero_grad()
    return loss.item()


@torch.no_grad()
def valid_step(model, x, y) -> float:
    target = y.transpose(0, 1)
    output = model(x).permute(1, 2, 0)
    loss = F.cross_entropy(output, target)
    return loss.item()

import torch
import torch.nn.functional as F

class TextGenerator:
    def __init__(self, model, vocab, device="cpu"):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device

    def _inp(self, indices: list[int]):
        """Preprocess indices (T,) to (T, 1, V) mini-batch shape with batch_size=1."""
        n = len(self.vocab)
        x = F.one_hot(torch.tensor(indices), n).float()
        return x.view(-1, 1, n).to(self.device)

    @staticmethod
    def sample_token(logits, temp: float):
        """Sample based on logits with softmax temperature."""
        # higher temp => more uniform, i.e. exp ~ 1
        p = F.softmax(logits / temp, dim=1)
        return torch.multinomial(p, num_samples=1).item()

    def predict(self, prompt: str, num_preds: int, temp=1.0):
        """Simulate character generation one at a time."""

        # Iterate over warmup text. RNN cell outputs final state
        warmup_indices = self.vocab[list(prompt.lower())]
        outs, state = self.model(self._inp(warmup_indices), return_state=True)

        # Next token sampling and state update
        indices = []
        for _ in range(num_preds):
            i = self.sample_token(logits=outs[-1], temp=temp)
            indices.append(i)
            outs, state = self.model(self._inp([i]), state, return_state=True)
        
        return "".join(self.vocab.to_tokens(warmup_indices + indices))

class LSTM(RNNBase):
    def __init__(self, inputs_dim: int, hidden_dim: int):
        super().__init__(inputs_dim, hidden_dim)
        self.I = nn.Linear(inputs_dim + hidden_dim, hidden_dim)
        self.F = nn.Linear(inputs_dim + hidden_dim, hidden_dim)
        self.O = nn.Linear(inputs_dim + hidden_dim, hidden_dim)
        self.G = nn.Linear(inputs_dim + hidden_dim, hidden_dim)

    def init_state(self, x):
        B = x.shape[1]
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        return h, c
    
    def _step(self, x_t, state):
        h, c = state
        x_gate = torch.cat([x_t, h], dim=1)
        g = torch.tanh(self.G(x_gate))
        i = torch.sigmoid(self.I(x_gate))
        f = torch.sigmoid(self.F(x_gate))
        o = torch.sigmoid(self.O(x_gate))
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, (h, c)

    def compute(self, x, state):
        T = x.shape[0]
        outs = []
        for t in range(T):
            out, state = self._step(x[t], state)
            outs.append(out)
        return torch.stack(outs), state

class GRU(RNNBase):
    def __init__(self, inputs_dim: int, hidden_dim: int):
        super().__init__(inputs_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.inputs_dim = inputs_dim
        self.R = nn.Linear(inputs_dim + hidden_dim, hidden_dim)
        self.Z = nn.Linear(inputs_dim + hidden_dim, hidden_dim)
        self.G = nn.Linear(inputs_dim + hidden_dim, hidden_dim)
    
    def init_state(self, x):
        B = x.shape[1]
        return torch.zeros(B, self.hidden_dim, device=x.device)

    def _step(self, x_t, state):
        h = state
        x_gate = torch.cat([x_t, h], dim=1)
        r = torch.sigmoid(self.R(x_gate))
        z = torch.sigmoid(self.Z(x_gate))
        g = torch.tanh(self.G(torch.cat([x_t, r * h], dim=1)))
        h = z * h + (1 - z) * g
        return h, h

    def compute(self, x, state):
        T = x.shape[0]
        outs = []
        for t in range(T):
            out, state = self._step(x[t], state)
            outs.append(out)
        return torch.stack(outs), state

from functools import partial

class DeepRNN(RNNBase):
    def __init__(self, 
        cell: Type[RNNBase],
        inputs_dim: int, hidden_dim: int,
        num_layers: int,    # (!)
        **kwargs,
    ):
        super().__init__(inputs_dim, hidden_dim)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            if l == 0:
                self.layers.append(cell(inputs_dim, hidden_dim, **kwargs))
            else:
                self.layers.append(cell(hidden_dim, hidden_dim, **kwargs))
  
    def init_state(self, x):
        """Defer state init to each cell with state=None."""
        return [None] * self.num_layers
    
    def compute(self, x, state):
        T = x.shape[0]
        out = x
        for l, cell in enumerate(self.layers):
            out, state[l] = cell(out, state[l])
        return out, state


Deep = lambda cell: partial(DeepRNN, cell)

from functools import partial

class BiRNN(RNNBase):
    def __init__(self, 
        cell: Type[RNNBase],
        inputs_dim: int, hidden_dim: int, 
        **kwargs
    ):
        super().__init__(inputs_dim, hidden_dim)
        assert hidden_dim % 2 == 0
        self.frnn = cell(inputs_dim, hidden_dim // 2, **kwargs)
        self.brnn = cell(inputs_dim, hidden_dim // 2, **kwargs)
        
    def init_state(self, x):
        return (None, None)

    def compute(self, x, state):
        fh, bh = state
        fo, fh = self.frnn(x, fh)
        bo, bh = self.brnn(torch.flip(x, [0]), bh)  # Flip seq index: (T, B, d)
        bo = torch.flip(bo, [0])                    # Flip back outputs. See above
        outs = torch.cat([fo, bo], dim=-1)
        return outs, (fh, bh)


Bidirectional = lambda cell: partial(BiRNN, cell)

