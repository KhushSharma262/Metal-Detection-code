import torch
import torch.nn as nn
from torchvision import models
from src.config import cfg

NUM_CLASSES = len(cfg.dataset.classes)

def build_model(num_classes=NUM_CLASSES, pretrained=True):
    weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
    backbone = models.efficientnet_b3(weights=weights)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=cfg.model.dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return backbone

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
