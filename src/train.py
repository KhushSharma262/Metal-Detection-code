import os
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from src.config import cfg
from src.model import build_model, freeze_backbone, unfreeze_backbone
from src.dataset_loader import get_dataloaders

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device     : {device}")
    print(f"[INFO] Classes    : {cfg.dataset.classes}")
    print(f"[INFO] Num classes: {len(cfg.dataset.classes)}")
    os.makedirs(cfg.paths.models_dir, exist_ok=True)
    os.makedirs(cfg.paths.logs_dir,   exist_ok=True)
    train_loader, val_loader, _, idx_to_class = get_dataloaders()
    model     = build_model(num_classes=len(cfg.dataset.classes), pretrained=True)
    model     = model.to(device)
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.scheduler_t_max)
    scaler    = GradScaler(enabled=(cfg.training.mixed_precision and device.type=="cuda"))
    best_val_acc   = 0.0
    patience_count = 0
    checkpoint     = None
    for epoch in range(1, cfg.training.epochs + 1):
        if epoch == cfg.model.freeze_backbone_epochs + 1:
            unfreeze_backbone(model)
            print(f"[INFO] Epoch {epoch}: backbone unfrozen")
        model.train()
        t_loss = t_correct = t_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(cfg.training.mixed_precision and device.type=="cuda")):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_loss    += loss.item() * imgs.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += labels.size(0)
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out    = model(imgs)
                loss   = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += (out.argmax(1) == labels).sum().item()
                v_total   += labels.size(0)
        ta = t_correct/t_total; tl = t_loss/t_total
        va = v_correct/v_total; vl = v_loss/v_total
        scheduler.step()
        print(f"Epoch {epoch:>3}/{cfg.training.epochs} | "
              f"Train {ta*100:.2f}% loss {tl:.4f} | "
              f"Val {va*100:.2f}% loss {vl:.4f}")
        if va > best_val_acc:
            best_val_acc   = va
            patience_count = 0
            checkpoint = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          va,
                "val_loss":         vl,
                "num_classes":      len(cfg.dataset.classes),
                "class_labels":     idx_to_class,
            }
            torch.save(checkpoint, cfg.paths.best_model)
            print(f"[Checkpoint] Saved -> {cfg.paths.best_model}  (val_acc={va:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.training.early_stopping_patience:
                print(f"[EarlyStopping] Triggered at epoch {epoch}.")
                break
    if checkpoint:
        torch.save(checkpoint, cfg.paths.final_model)
    print("="*60)
    print(f" Done! Best val_acc = {best_val_acc:.4f}")
    print(f" Model saved to    : {cfg.paths.best_model}")
    print("="*60)

if __name__ == "__main__":
    train()
