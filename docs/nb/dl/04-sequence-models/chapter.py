import re
from collections import Counter
from typing import Union, Optional, TypeVar, List

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

import torch

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

import re
import os
import requests

from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


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
        return encoded_text, tokenizer, vocab

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
from typing import List
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, 
        names: List[str], 
        block_size: int,
        vocab: Optional[Vocab] = None
    ):
        self.block_size = block_size
        self.vocab = vocab or Vocab(text="".join(names), preprocess=False, reserved_tokens=[PAD_TOKEN])
        self.tokenizer = Tokenizer(self.vocab)
        self.xs, self.ys = self.samples(names)

    def vocab_size(self):
        return len(self.vocab)
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, i: int):
        x = self.tokenizer.encode(self.xs[i])
        y = self.tokenizer.encode(self.ys[i])[0]
        return x, y

    def samples(self, names: List[str]):
        xs, ys = [], []
        for name in names:
            context = PAD_TOKEN * self.block_size
            for c in name + PAD_TOKEN:
                xs.append(context)
                ys.append(c)
                context = context[1:] + c
        return xs, ys

class CountingModel:
    def __init__(self, block_size: int, vocab_size: int, alpha=0.01):
        """Model of observed n-grams to estimate next char probability."""
        self.P = None                    # cond. prob
        self.N = None                    # counts
        self.alpha = alpha               # laplace smoothing
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __call__(self, x: torch.tensor) -> torch.tensor:
        # tuple(x.T) = ([x11, x21, x31], [x12, x22, x32]) 
        # i.e. len = block_size, num entries = B
        # then, P[tuple(x.T)][b] == P[xb1, xb2], so output has shape (B, vocab_size) 
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
        x = dataset.tokenizer.encode(context).view(1, -1)
        p = model(x)[0]
        j = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        c = dataset.tokenizer.decode(j)
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
    
    def run(self, epochs, train_loader, valid_loader):
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
                w = int(0.05 * steps_per_epoch)
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

import re
from collections import Counter
from typing import Union, Optional, TypeVar, List

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

import torch

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

import re
import os
import requests

from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


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

import re
from collections import Counter
from typing import Union, Optional, TypeVar, List

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

import torch

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

import re
import os
import requests

from pathlib import Path

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


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

