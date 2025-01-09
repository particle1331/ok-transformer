import re
from collections import Counter
from typing import Union, Optional


class Vocab:
    def __init__(self, 
        text: str, 
        min_freq: int = 0, 
        reserved_tokens: Optional[list[str]] = None
    ):
        tokens = list(self.preprocess(text))    # character-level on clean text
        counter = Counter(tokens)
        reserved_tokens = reserved_tokens or []
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.itos = [self.unk_token] + reserved_tokens + [tok for tok, f in filter(lambda tokf: tokf[1] >= min_freq, self.token_freqs)]
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, tokens: Union[str, list[str]]) -> Union[int, list[int]]:
        if isinstance(tokens, str):
            return self.stoi.get(tokens, self.unk)
        else:
            return [self.__getitem__(tok) for tok in tokens]
            
    def to_tokens(self, indices: Union[int, list[int]]) -> Union[str, list[str]]:
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

class Tokenizer:
    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def tokenize(self, text: str) -> list[str]:
        return [tok if tok in self.vocab.stoi else self.vocab.unk_token for tok in list(text)]

    def encode(self, text: str) -> list[int]:
        return self.vocab[self.tokenize(text)]

    def decode(self, indices: Union[int, list[int]]) -> str:
        return "".join(self.vocab.to_tokens(indices))

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
        corpus = tokenizer.encode(vocab.preprocess(self.text))
        return corpus, tokenizer, vocab

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

