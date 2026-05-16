from __future__ import annotations
import csv
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from src.config import cfg
from src.dataset_loader import build_dataloaders
from src.model import build_model, save_checkpoint


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[Device] GPU : {torch.cuda.get_device_name(0)}")
    else:
        print("[Device] CPU only")
    return device


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )


def build_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.scheduler_t_max
    )


def compute_class_weights(dataset, num_classes, device):
    from collections import Counter
    counts  = Counter(dataset.targets)
    total   = sum(counts.values())
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_epoch(model, loader, criterion, optimizer, scaler,
              device, phase, use_amp, print_every):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    running_loss = correct = total = 0

    with torch.set_grad_enabled(is_train):
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train and use_amp:
                with autocast():
                    logits = model(images)
                    loss   = criterion(logits, labels)
            else:
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            if is_train and (batch_idx + 1) % print_every == 0:
                print(f"  [{phase}] batch {batch_idx+1:>3}/{len(loader)} "
                      f"loss={loss.item():.4f}  "
                      f"acc={(preds==labels).float().mean().item():.4f}")

    return running_loss / total, correct / total


class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter  = 0
        self.best     = -float("inf")
        self.stop     = False

    def step(self, metric):
        if metric > self.best:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class CSVLogger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch","train_loss","train_acc","val_loss","val_acc","lr"])
        self.path = path

    def log(self, epoch, tl, ta, vl, va, lr):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tl:.6f}", f"{ta:.6f}",
                 f"{vl:.6f}", f"{va:.6f}", f"{lr:.8f}"])


def train():
    set_seed()

    models_dir = Path(cfg.paths.models_dir)
    logs_dir   = Path(cfg.paths.logs_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _, class_names = build_dataloaders()
    num_classes = len(class_names)

    device  = get_device()
    use_amp = cfg.training.mixed_precision and device.type == "cuda"
    scaler  = GradScaler() if use_amp else None
    print(f"[AMP] Mixed precision: {use_amp}")

    model = build_model(num_classes, device)

    weights   = compute_class_weights(train_loader.dataset.dataset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    print(f"[Loss] Class weights: {weights.cpu().numpy().round(3)}")

    freeze_epochs = cfg.model.freeze_backbone_epochs
    model.freeze_backbone()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    writer     = SummaryWriter(log_dir=str(logs_dir / "tensorboard"))
    csv_logger = CSVLogger(str(logs_dir / "training_log.csv"))
    early_stop = EarlyStopping(patience=cfg.training.early_stopping_patience)
    best_val_acc = 0.0
    print_every  = cfg.logging.print_every_n_batches

    print(f"\n{'='*60}")
    print(f" Training  |  {num_classes} classes  |  {cfg.training.epochs} epochs")
    print(f" Classes   : {class_names}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()

        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = build_optimizer(model)
            scheduler = build_scheduler(optimizer)
            print(f"[Epoch {epoch}] Full fine-tuning started")

        print(f"\nEpoch {epoch}/{cfg.training.epochs}  "
              f"LR={optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, "train", use_amp, print_every)

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None,
            None, device, "val", use_amp, print_every)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"  -> train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"time={elapsed:.1f}s")

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        csv_logger.log(epoch, train_loss, train_acc, val_loss, val_acc,
                       optimizer.param_groups[0]["lr"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc,
                            class_names, cfg.paths.best_model)

        if early_stop.step(val_acc):
            print(f"\n[EarlyStopping] Triggered at epoch {epoch}.")
            break

    save_checkpoint(model, optimizer, epoch, val_acc,
                    class_names, cfg.paths.final_model)
    writer.close()

    print(f"\n{'='*60}")
    print(f" Done! Best val_acc = {best_val_acc:.4f}")
    print(f" Model saved to    : {cfg.paths.best_model}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
