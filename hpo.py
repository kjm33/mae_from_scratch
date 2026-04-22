"""
Hyperparameter search for MAE using Optuna + PyTorch Lightning.
Single-GPU, short trials — intended to narrow down good ranges before a full run.

Searched hyperparameters:
  - optimizer: AdamW, Adam, AdaBelief, RAdam, Lion, MADGRAD, SGD
  - lr, weight_decay, mask_ratio
  - betas (Adam-family: AdamW/Adam/AdaBelief/RAdam/Lion)
  - momentum (SGD-family: SGD/MADGRAD)

Usage:
    python hpo.py
    python hpo.py --n-trials 50 --n-epochs 15 --device 1
    python hpo.py --storage sqlite:///hpo.db   # persistent study — safe to Ctrl-C and resume
"""

import argparse
import os

import numpy as np
import optuna
import torch
import torch_optimizer
import lion_pytorch
import lightning.pytorch as pl
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader, Dataset

from mae import mae_vit_small_patch32x8

torch.set_float32_matmul_precision("high")

# Optimizers grouped by hyperparameter signature.
# betas_family: (lr, weight_decay, beta1, beta2)
# momentum_family: (lr, weight_decay, momentum)
BETAS_OPTIMIZERS    = ("adamw", "adam", "adabelief", "radam", "lion")
MOMENTUM_OPTIMIZERS = ("sgd", "madgrad")

# Suppress Lightning / TF boilerplate
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LIGHTNING_DISABLE_HINTS", "1")

NPY_PATH     = "./data/yiddish_lines.npy"
BATCH_SIZE   = 256
NUM_WORKERS  = 4
VAL_FRACTION = 0.05


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _MAEDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32) / 255.0  # (H, W)
        return torch.from_numpy(img).unsqueeze(0)          # (1, H, W)


def _make_loaders():
    data  = np.load(NPY_PATH, mmap_mode="r")
    n     = len(data)
    val_n = max(1, int(n * VAL_FRACTION))
    train_ds = _MAEDataset(data[: n - val_n])
    val_ds   = _MAEDataset(data[n - val_n :])
    shared = dict(num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True,  **shared),
        DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False, **shared),
    )


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class _EpochPrinter(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch     = trainer.current_epoch + 1
        max_epoch = trainer.max_epochs
        train_loss = trainer.callback_metrics.get("train_loss", float("nan"))
        print(f"  [{epoch:>2d}/{max_epoch}]  train={float(train_loss):.4f}", end="  ", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss", float("nan"))
        print(f"val={float(val_loss):.4f}", flush=True)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class MAEModule(pl.LightningModule):
    def __init__(self, optimizer_name, optimizer_kwargs,
                 lr, weight_decay, mask_ratio, n_epochs, steps_per_epoch):
        super().__init__()
        self.save_hyperparameters()
        self.model = mae_vit_small_patch32x8()

    def _step(self, batch):
        x = batch
        latent, mask, ids_restore = self.model.forward_encoder(x, mask_ratio=self.hparams.mask_ratio)
        pred = self.model.forward_decoder(latent, ids_restore)
        return self.model.forward_loss(x, pred, mask)

    def training_step(self, batch, _):
        loss = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        hp   = self.hparams
        name = hp.optimizer_name
        kw   = dict(hp.optimizer_kwargs)   # copy — may contain betas tuple as list

        # Restore tuple that save_hyperparameters serialises as a list
        if "betas" in kw:
            kw["betas"] = tuple(kw["betas"])

        base = dict(params=self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        if name == "adamw":
            opt = torch.optim.AdamW(**base, **kw)
        elif name == "adam":
            opt = torch.optim.Adam(**base, **kw)
        elif name == "adabelief":
            opt = torch_optimizer.AdaBelief(**base, **kw)
        elif name == "radam":
            opt = torch_optimizer.RAdam(**base, **kw)
        elif name == "lion":
            opt = lion_pytorch.Lion(**base, **kw)
        elif name == "madgrad":
            opt = torch_optimizer.MADGRAD(**base, **kw)
        else:  # sgd
            opt = torch.optim.SGD(**base, **kw)

        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=hp.lr,
            steps_per_epoch=hp.steps_per_epoch,
            epochs=hp.n_epochs,
            pct_start=0.05,
            anneal_strategy="cos",
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, args) -> float:
    # --- shared hyperparameters ---
    choices = [args.optimizer] if args.optimizer else list(BETAS_OPTIMIZERS + MOMENTUM_OPTIMIZERS)
    optimizer_name = trial.suggest_categorical("optimizer", choices)
    # Lion works best at ~1/10 AdamW lr, so widen the range to let Optuna discover this.
    lr           = trial.suggest_float("lr",           1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 0.1,  log=True)
    mask_ratio   = trial.suggest_float("mask_ratio",   0.55, 0.80)

    # --- optimizer-specific hyperparameters ---
    if optimizer_name in BETAS_OPTIMIZERS:
        beta1 = trial.suggest_float("beta1", 0.85, 0.95)
        beta2 = trial.suggest_float("beta2", 0.90, 0.999)
        optimizer_kwargs = {"betas": (beta1, beta2)}
        opt_info = f"β=({beta1:.3f},{beta2:.4f})"
    else:  # momentum family
        momentum = trial.suggest_float("momentum", 0.80, 0.99)
        optimizer_kwargs = {"momentum": momentum}
        if optimizer_name == "sgd":
            optimizer_kwargs["nesterov"] = True
        opt_info = f"momentum={momentum:.3f}"

    print(f"\nTrial {trial.number:>3d} | {optimizer_name:<10s} lr={lr:.2e}  wd={weight_decay:.2e}  "
          f"mask={mask_ratio:.2f}  {opt_info}", flush=True)

    train_loader, val_loader = _make_loaders()

    module = MAEModule(
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        lr=lr,
        weight_decay=weight_decay,
        mask_ratio=mask_ratio,
        n_epochs=args.n_epochs,
        steps_per_epoch=len(train_loader),
    )

    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator="gpu",
        devices=[args.device],
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss"), _EpochPrinter()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf"))).item()
    print(f"  → val_loss={val_loss:.4f}  {'[PRUNED]' if trainer.should_stop else ''}", flush=True)
    return val_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAE hyperparameter search (Optuna + Lightning)")
    parser.add_argument("--n-trials",   type=int, default=30,  help="Number of Optuna trials")
    parser.add_argument("--n-epochs",   type=int, default=8,   help="Training epochs per trial")
    parser.add_argument("--device",     type=int, default=0,   help="GPU index")
    parser.add_argument("--study-name", default="mae_hpo")
    parser.add_argument("--storage",    default=None,          help="e.g. sqlite:///hpo.db — enables resume")
    parser.add_argument("--optimizer",  default=None,          choices=list(BETAS_OPTIMIZERS + MOMENTUM_OPTIMIZERS),
                        help="Fix optimizer for phase-2 fine search (default: search all)")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study  = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage,
        load_if_exists=True,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print(f"\n=== Best trial (val_loss={study.best_trial.value:.4f}) ===")
    for k, v in study.best_trial.params.items():
        print(f"  {k:15s}: {v:.6g}" if isinstance(v, float) else f"  {k:15s}: {v}")


if __name__ == "__main__":
    main()
