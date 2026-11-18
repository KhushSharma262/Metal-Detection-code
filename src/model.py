import torch
import torch.nn as nn
from torchvision import models
from src.config import cfg

SUPPORTED = ("convnext_tiny", "efficientnet_v2_s", "efficientnet_b3")


def build_model(num_classes, architecture=None, pretrained=True, dropout=None):
    arch = architecture or cfg.model.architecture
    dropout = cfg.model.dropout if dropout is None else dropout
    if arch == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        m = models.convnext_tiny(weights=w)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif arch == "efficientnet_v2_s":
        w = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_v2_s(weights=w)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif arch == "efficientnet_b3":
        w = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b3(weights=w)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED}")
    return m


def freeze_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier")


def unfreeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = True


def set_train_mode(model, backbone_frozen):
    model.train()
    if backbone_frozen:
        model.features.eval()


def get_param_groups(model, head_lr, backbone_lr, weight_decay):
    buckets = {}
    for name, p in model.named_parameters():
        is_head = name.startswith("classifier")
        no_decay = p.ndim <= 1
        buckets.setdefault((is_head, no_decay), []).append(p)
    groups = []
    for (is_head, no_decay), params in buckets.items():
        groups.append({
            "params": params,
            "lr": head_lr if is_head else backbone_lr,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "name": ("head" if is_head else "backbone") + ("_nodecay" if no_decay else ""),
        })
    return groups


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    classes = list(ckpt["class_names"])
    model = build_model(len(classes), architecture=ckpt["architecture"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt