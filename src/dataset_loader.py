from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import Image
from src.config import cfg
from src.augmentations import get_train_transforms, get_val_transforms


class MetalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self._base = ImageFolder(root=str(self.root_dir))
        self.classes = self._base.classes
        self.class_to_idx = self._base.class_to_idx
        self.samples = self._base.samples
        self.targets = self._base.targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def stratified_split(dataset, train_ratio=0.70, val_ratio=0.15, seed=42):
    indices = list(range(len(dataset)))
    labels  = dataset.targets
    test_ratio = 1.0 - train_ratio - val_ratio
    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels, random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - relative_val),
        stratify=temp_labels, random_state=seed,
    )
    return train_idx, val_idx, test_idx


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_dataloaders(
    data_dir=cfg.dataset.root_dir,
    batch_size=cfg.training.batch_size,
    num_workers=cfg.training.num_workers,
    pin_memory=cfg.training.pin_memory,
):
    full_dataset = MetalDataset(root_dir=data_dir)
    class_names  = full_dataset.classes
    print(f"[Dataset] Classes found : {class_names}")
    print(f"[Dataset] Total samples : {len(full_dataset)}")
    train_idx, val_idx, test_idx = stratified_split(full_dataset)
    print(f"[Split]   Train : {len(train_idx)} samples")
    print(f"[Split]   Val   : {len(val_idx)} samples")
    print(f"[Split]   Test  : {len(test_idx)} samples")
    train_set = TransformSubset(full_dataset, train_idx, get_train_transforms())
    val_set   = TransformSubset(full_dataset, val_idx,   get_val_transforms())
    test_set  = TransformSubset(full_dataset, test_idx,  get_val_transforms())
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_set,  shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, class_names
