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
