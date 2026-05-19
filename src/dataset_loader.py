import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.config import cfg

def get_transforms(split="train"):
    size = cfg.dataset.image_size
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=tuple(cfg.augmentation.random_resized_crop_scale)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(cfg.augmentation.random_rotation_degrees),
            transforms.ColorJitter(
                brightness=cfg.augmentation.brightness,
                contrast=cfg.augmentation.contrast,
                saturation=cfg.augmentation.saturation,
                hue=cfg.augmentation.hue,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def get_dataloaders():
    root         = cfg.dataset.root_dir
    full_dataset = datasets.ImageFolder(root, transform=get_transforms("train"))
    labels_path  = cfg.paths.class_labels
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    with open(labels_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"[INFO] Class labels saved -> {labels_path}")
    print(f"[INFO] Index mapping: {idx_to_class}")
    total   = len(full_dataset)
    train_n = int(total * cfg.dataset.train_split)
    val_n   = int(total * cfg.dataset.val_split)
    test_n  = total - train_n - val_n
    generator = torch.Generator().manual_seed(cfg.dataset.random_seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_n, val_n, test_n], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader, idx_to_class
