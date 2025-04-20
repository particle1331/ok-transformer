import torch
import numpy as np
from tqdm import tqdm
from okt.nn.utils import get_device, eval_context

import matplotlib.pyplot as plt


DEVICE = get_device()


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
        with eval_context(self.model):
            preds, y = self.forward(batch)
            accs = (preds.argmax(dim=1) == y).float().sum()
            loss = self.loss_fn(preds, y, reduction="sum")
        return {"loss": loss, "accs": accs}
    
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

    def plot_training_history(
        self, accs_offset=0.05, loss_offset=-0.1, figsize=(8, 4), annotate=False, markersize=12
    ):
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        num_epochs = len(self.valid_log["accs"])
        num_steps_per_epoch = len(self.train_log["accs"]) // num_epochs
        xv = list(range(num_steps_per_epoch, (num_epochs + 1) * num_steps_per_epoch, num_steps_per_epoch))
        
        ax[0].plot(self.train_log["loss"], alpha=0.3, color="C0")
        ax[1].plot(self.train_log["accs"], alpha=0.3, color="C0")
        ax[0].plot(self.train_log["loss_avg"], label="train", color="C0")
        ax[1].plot(self.train_log["accs_avg"], label="train", color="C0")
        
        ax[0].plot(xv, self.valid_log["loss"], label="valid", color="C1", linestyle="--")
        for x, y in zip(xv, self.valid_log["loss"]):
            ax[0].scatter(x, y, edgecolor="black", facecolor="orange", marker="v", s=markersize, zorder=5)
            if annotate:
                ax[0].text(x, y + loss_offset, f"{y:.2f}", ha="center", fontsize=9, zorder=5)

        ax[1].plot(xv, self.valid_log["accs"], label="valid", color="C1", linestyle="--")
        for x, y in zip(xv, self.valid_log["accs"]):
            ax[1].scatter(x, y, edgecolor="black", facecolor="orange", marker="v", s=markersize, zorder=5)
            if annotate:
                ax[1].text(x, y + accs_offset, f"{y:.2f}", ha="center", fontsize=9, zorder=5)

        ax[0].set_xlabel("step")
        ax[0].set_ylabel("loss")
        ax[0].grid(linestyle="dashed", alpha=0.3)
        ax[1].set_xlabel("step")
        ax[1].set_ylabel("accuracy")
        ax[1].grid(linestyle="dashed", alpha=0.3)
        ax[1].legend()
        ax[0].set_ylim(0, max(self.train_log["loss"]))
        ax[1].set_ylim(0, 1)
        ax[0].ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        ax[1].ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        
        fig.tight_layout()
        return fig, ax
