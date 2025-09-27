import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
from torchmetrics.classification import (
    MulticlassCohenKappa, MulticlassConfusionMatrix
)

import pytorch_lightning as pl
from utils.lr_scheduler import linear_warmup_cosine_decay
import random

# Helper: Randomly selects a subset of EEG channels (augmentations)
def select_random_channels(x, keep_ratio=0.9):
    """
    Select a subset of EEG channels.
    Args:
        x: Tensor of shape [B, C, T]
        keep_ratio: fraction of channels to keep
    Returns:
        Tensor of shape [B, C_selected, T]
    """
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    return x[:, keep_indices, :], keep_indices

# Helper: Randomly masks EEG channels (augmentations)
def random_channel_mask(x, keep_ratio=0.9):
    """
    Randomly keeps a subset of EEG channels.
    Args:
        x: Tensor of shape [B, C, T]
        keep_ratio: Float, ratio of channels to keep (e.g., 0.9 to keep 90%).
    Returns:
        Augmented tensor with masked channels set to 0.
    """
    B, C, T = x.shape
    keep_chs = int(C * keep_ratio)
    keep_indices = sorted(random.sample(range(C), keep_chs))
    mask = torch.zeros_like(x)
    mask[:, keep_indices, :] = 1
    return x * mask

# Lightning module
class ClassificationModule(pl.LightningModule):
    def __init__(
            self,
            model,
            n_classes,
            lr=0.001,
            weight_decay=0.0,
            optimizer="adam",
            scheduler=False,
            max_epochs=1000,
            warmup_epochs=20,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        # ── metrics ───────────────────────────────────────
        self.test_kappa = MulticlassCohenKappa(num_classes=n_classes)        
        self.test_cm = MulticlassConfusionMatrix(num_classes=n_classes)  
        # will hold the final cm on CPU after test
        self.test_confmat = None

    # forward
    def forward(self, x):
        return self.model(x)

    # optimiser / scheduler
    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                          betas=betas,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(optimizer,
                                 linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                                            self.hparams.max_epochs))
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    # steps
    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="val")
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch, batch_idx, mode="test")
        return {"test_loss": loss, "test_acc": acc}

    # common logic
    def shared_step(self, batch, batch_idx, mode: str = "train"):
        x, y = batch
        if mode == "train":
            if self.hparams.get("random_channel_masking", False):
                # Add random EEG channel masking augmentation (did not improve the training)
                x = random_channel_mask(x, self.hparams.get("keep_ratio",0.9))
            if self.hparams.get("random_channel_selection", False):
                # Randomly select a subset of EEG channels (did not improve the training)
                x, _ = select_random_channels(x, self.hparams.get("keep_ratio",0.9))

        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
        # log scalar metrics
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        if mode == "test":
            preds = torch.argmax(y_hat, dim=-1)
            # ── update epoch-level metrics ───────────────────────────
            self.test_kappa.update(preds, y)                       # accumulate
            self.test_cm.update(preds, y)
            
            self.log("test_kappa", self.test_kappa,                # Lightning will call .compute()
                    prog_bar=False, on_step=False, on_epoch=True)

        return loss, acc
    
    # grab confusion matrix once per test epoch
    def on_test_epoch_end(self):
        # 1) raw counts  ───────────────────────────────────────────
        cm_counts = self.test_cm.compute()   # shape [C, C]
        self.test_cm.reset()

        # 2) row-normalise → %  (handle rows with 0 samples safely)
        with torch.no_grad():
            row_sums = cm_counts.sum(dim=1, keepdim=True).clamp_min(1)
            cm_percent = cm_counts.float() / row_sums * 100.0

        self.test_confmat = cm_percent.cpu()        # stash for plotting

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)
    