import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib_inline import backend_inline

import torch
import torch.nn as nn
import torch.nn.functional as F

DATASET_DIR = Path("./data/").resolve()
DATASET_DIR.mkdir(exist_ok=True)
warnings.simplefilter(action="ignore")
backend_inline.set_matplotlib_formats("svg")
matplotlib.rcParams["image.interpolation"] = "nearest"

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

import torchsummary

mnist_model = lambda: nn.Sequential(
    nn.Conv2d(1, 32, 3, 1, 1),
    nn.SELU(),
    nn.MaxPool2d(2, 2),
    
    nn.Conv2d(32, 32, 5, 1, 0),
    nn.SELU(),
    nn.MaxPool2d(2, 2),
    
    nn.Flatten(),
    nn.Linear(800, 256), nn.SELU(), nn.Dropout(0.5),
    nn.Linear(256, 10)
)

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.)
])

g = torch.Generator().manual_seed(RANDOM_SEED)
ds = MNIST(root=DATASET_DIR, download=False, transform=transform)
ds_train, ds_valid = random_split(ds, [55000, 5000], generator=g)
dl_train = DataLoader(ds_train, batch_size=32, shuffle=True) # (!)
dl_valid = DataLoader(ds_valid, batch_size=32, shuffle=False)

from tqdm.notebook import tqdm
from contextlib import contextmanager
from torch.utils.data import DataLoader


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
        self.train_log = {"loss": [], "accs": [], "loss_avg": [], "accs_avg": []}
        self.valid_log = {"loss": [], "accs": []}
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
        accs = (preds.argmax(dim=1) == y).float().mean()
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return {"loss": loss, "accs": accs}

    @torch.inference_mode()
    def valid_step(self, batch):
        preds, y = self.forward(batch)
        accs = (preds.argmax(dim=1) == y).float().sum()
        loss = self.loss_fn(preds, y, reduction="sum")
        return {"loss": loss, "accs": accs}
    
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
                self.train_log["accs"].append(output["accs"].item())
                self.train_log["loss_avg"].append(np.mean(self.train_log["loss"][-w:]))
                self.train_log["accs_avg"].append(np.mean(self.train_log["accs"][-w:]))

            # logs @ epoch
            output = self.evaluate(valid_loader)
            self.valid_log["loss"].append(output["loss"])
            self.valid_log["accs"].append(output["accs"])
            if self.verbose:
                print(f"[Epoch: {e+1:>0{int(len(str(epochs)))}d}/{epochs}]    loss: {self.train_log['loss_avg'][-1]:.4f}  acc: {self.train_log['accs_avg'][-1]:.4f}    val_loss: {self.valid_log['loss'][-1]:.4f}  val_acc: {self.valid_log['accs'][-1]:.4f}")

    def evaluate(self, data_loader):
        with eval_context(self.model):
            valid_loss = 0.0
            valid_accs = 0.0
            for batch in data_loader:
                output = self.valid_step(batch)
                valid_loss += output["loss"].item()
                valid_accs += output["accs"].item()

        return {
            "loss": valid_loss / len(data_loader.dataset),
            "accs": valid_accs / len(data_loader.dataset)
        }

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        with eval_context(self.model):
            return self(x)

from torch.optim.lr_scheduler import OneCycleLR

class SchedulerStatsCallback:
    def __init__(self, optim):
        self.lr = []
        self.momentum = []
        self.optim = optim

    def __call__(self):
        self.lr.append(self.optim.param_groups[0]["lr"])
        self.momentum.append(self.optim.param_groups[0]["betas"][0])

epochs = 3
model = mnist_model().to(DEVICE)
loss_fn = F.cross_entropy
optim = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = OneCycleLR(optim, max_lr=0.01, steps_per_epoch=len(dl_train), epochs=epochs)
scheduler_stats = SchedulerStatsCallback(optim)
trainer = Trainer(model, optim, loss_fn, scheduler, callbacks=[scheduler_stats])

import cv2

IMG_DATASET_DIR = DATASET_DIR / "histopathologic-cancer-detection"
data = pd.read_csv(IMG_DATASET_DIR / "train_labels.csv")

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.CenterCrop([49, 49]),
])

transform_infer = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop([49, 49]),
])

from torch.utils.data import DataLoader, Dataset, Subset

class HistopathologicDataset(Dataset):
    def __init__(self, data, train=True, transform=None):
        split = "train" if train else "test"
        self.fnames = [str(IMG_DATASET_DIR / split / f"{fn}.tif") for fn in data.id]
        self.labels = data.label.tolist()
        self.transform = transform
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        img = cv2.imread(self.fnames[index])
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[index]


data = data.sample(frac=1.0)
split = int(0.80 * len(data))
ds_train = HistopathologicDataset(data[:split], train=True, transform=transform_train)
ds_valid = HistopathologicDataset(data[split:], train=True, transform=transform_infer)

def plot_training_history(trainer):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    num_epochs = len(trainer.valid_log["accs"])
    num_steps_per_epoch = len(trainer.train_log["accs"]) // num_epochs
    ax[0].plot(trainer.train_log["loss"], alpha=0.3, color="C0")
    ax[1].plot(trainer.train_log["accs"], alpha=0.3, color="C0")
    ax[0].plot(trainer.train_log["loss_avg"], label="train", color="C0")
    ax[1].plot(trainer.train_log["accs_avg"], label="train", color="C0")
    ax[0].plot(list(range(num_steps_per_epoch, (num_epochs + 1) * num_steps_per_epoch, num_steps_per_epoch)), trainer.valid_log["loss"], label="valid", color="C1")
    ax[1].plot(list(range(num_steps_per_epoch, (num_epochs + 1) * num_steps_per_epoch, num_steps_per_epoch)), trainer.valid_log["accs"], label="valid", color="C1")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("loss")
    ax[0].grid(linestyle="dashed", alpha=0.3)
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("accuracy")
    ax[1].grid(linestyle="dashed", alpha=0.3)
    ax[1].legend()
    ax[0].set_ylim(0, max(trainer.train_log["loss"]))
    ax[1].set_ylim(0, 1)
    ax[0].ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    ax[1].ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    
    fig.tight_layout();

