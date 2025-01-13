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
    def __init__(self, data: torch.Tensor, seq_len: int, vocab_size: int):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __getitem__(self, i):
        c = self.data[i: i + self.seq_len + 1]
        x, y = c[:-1], c[1:]
        x = F.one_hot(x, num_classes=self.vocab_size).float()
        return x, y
    
    def __len__(self):
        return len(self.data) - self.seq_len

import re
import os
import torch
import requests
from collections import Counter
from typing import Union, Optional, TypeVar, List

from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


T = TypeVar("T")
ScalarOrList = Union[T, List[T]]


class Vocab:
    def __init__(self, 
        text: str, 
        min_freq: int = 0, 
        reserved_tokens: Optional[List[str]] = None,
        preprocess: bool = True
    ):
        text = self.preprocess(text) if preprocess else text
        tokens = list(text)
        counter = Counter(tokens)
        reserved_tokens = reserved_tokens or []
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.itos = [self.unk_token] + reserved_tokens + [tok for tok, f in filter(lambda tokf: tokf[1] >= min_freq, self.token_freqs)]
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, tokens: ScalarOrList[str]) -> ScalarOrList[int]:
        if isinstance(tokens, str):
            return self.stoi.get(tokens, self.unk)
        else:
            return [self.__getitem__(tok) for tok in tokens]

    def to_tokens(self, indices: ScalarOrList[int]) -> ScalarOrList[str]:
        if isinstance(indices, int):
            return self.itos[indices]
        else:
            return [self.itos[int(index)] for index in indices]
            
    def preprocess(self, text: str):
        return re.sub("[^A-Za-z]+", " ", text).lower().strip()

    @property
    def unk_token(self) -> str:
        return "▮"

    @property
    def unk(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def tokens(self) -> List[int]:
        return self.itos


class Tokenizer:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def tokenize(self, text: str) -> List[str]:
        UNK = self.vocab.unk_token
        tokens = self.vocab.stoi.keys()
        return [c if c in tokens else UNK for c in list(text)]

    def encode(self, text: str) -> torch.Tensor:
        x = self.vocab[self.tokenize(text)]
        return torch.tensor(x, dtype=torch.int64)

    def decode(self, indices: Union[ScalarOrList[int], torch.Tensor]) -> str:
        return "".join(self.vocab.to_tokens(indices))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class TimeMachine:
    def __init__(self, download=False, path=None):
        DEFAULT_PATH = str((DATA_DIR / "time_machine.txt").absolute())
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
        
    def _load_text(self):
        with open(self.filepath, "r") as f:
            text = f.read()
        s = "*** START OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***"
        e = "*** END OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***"
        return text[text.find(s) + len(s): text.find(e)]
    
    def build(self, vocab: Optional[Vocab] = None):
        self.text = self._load_text()
        vocab = vocab or Vocab(self.text)
        tokenizer = Tokenizer(vocab)
        encoded_text = tokenizer.encode(vocab.preprocess(self.text))
        return encoded_text, tokenizer

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
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer

    def _inp(self, indices: list[int]):
        """Preprocess indices (T,) to (T, 1, V) shape with B=1."""
        VOCAB_SIZE = self.tokenizer.vocab_size
        x = F.one_hot(torch.tensor(indices), VOCAB_SIZE).float()
        return x.view(-1, 1, VOCAB_SIZE).to(self.device)

    @staticmethod
    def sample_token(logits, temperature: float):
        """Convert logits to probs with softmax temperature."""
        p = F.softmax(logits / temperature, dim=1)  # T = ∞ => exp ~ 1 => p ~ U[0, 1]
        return torch.multinomial(p, num_samples=1).item()

    def predict(self, prompt: str, num_preds: int, temperature=1.0):
        """Simulate character generation one at a time."""

        # Iterate over warmup text. RNN cell outputs final state
        warmup_indices = self.tokenizer.encode(prompt.lower()).tolist()
        outs, state = self.model(self._inp(warmup_indices), return_state=True)

        # Sample next token and update state
        indices = []
        for _ in range(num_preds):
            i = self.sample_token(outs[-1], temperature)
            indices.append(i)
            outs, state = self.model(self._inp([i]), state, return_state=True)

        return self.tokenizer.decode(warmup_indices + indices)

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

