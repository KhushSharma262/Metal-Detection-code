from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from src.config import cfg


class MetalClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        weights  = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        self.features   = backbone.features
        self.avgpool    = backbone.avgpool
        in_features     = backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2, inplace=False),
            nn.Linear(512, num_classes),
        )
        print(f"[Model] MetalClassifier-B3 | classes={num_classes} | "
              f"pretrained={pretrained} | dropout={dropout}")

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
        print("[Model] Backbone FROZEN")

    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True
        print("[Model] Backbone UNFROZEN")

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(num_classes, device):
    model  = MetalClassifier(num_classes=num_classes)
    model  = model.to(device)
    counts = model.count_parameters()
    print(f"[Model] Total params     : {counts['total']:,}")
    print(f"[Model] Trainable params : {counts['trainable']:,}")
    print(f"[Model] Device           : {device}")
    return model


def save_checkpoint(model, optimizer, epoch, val_acc, class_names, path):
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"       : epoch,
        "val_acc"     : val_acc,
        "num_classes" : model.num_classes,
        "class_names" : class_names,
        "model_state" : model.state_dict(),
        "optim_state" : optimizer.state_dict(),
    }, path)
    print(f"[Checkpoint] Saved -> {path}  (epoch={epoch}, val_acc={val_acc:.4f})")


def load_checkpoint(path, device):
    checkpoint  = torch.load(path, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    model       = MetalClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    print(f"[Checkpoint] Loaded <- {path}")
    print(f"[Checkpoint] Classes: {checkpoint['class_names']}")
    return model, checkpoint
