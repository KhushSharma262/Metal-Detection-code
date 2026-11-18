"""Grad-CAM for the metal classifier (works with torchvision ConvNeXt / EfficientNet).

Usage:
    with GradCAM(model) as g:
        heatmap = g.generate(x)                  # predicted class
        heatmap = g.generate(x, class_idx=5)     # a specific class
Heatmap: float32 array (H, W) in [0, 1], same H, W as the input tensor.
This is an interpretability aid, not proof of what the model "used".
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self._activations = None
        self._gradients = None
        self.last_class_idx = None
        self.last_probs = None
        self.last_cam_empty = False
        self._handle = self.target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self._activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._on_grad)

    def _on_grad(self, grad):
        self._gradients = grad.detach()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False

    def generate(self, image_tensor, class_idx=None):
        if self._handle is None:
            raise RuntimeError("GradCAM hooks were removed; create a new GradCAM.")
        x = image_tensor.detach()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError(f"Expected one image of shape (3, H, W) or (1, 3, H, W); got {tuple(image_tensor.shape)}")

        was_training = self.model.training
        device = next(self.model.parameters()).device
        x = x.to(device).float().requires_grad_(True)
        self._activations, self._gradients = None, None
        self.model.eval()
        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                logits = self.model(x)
                n_classes = logits.shape[1]
                if class_idx is None:
                    idx = int(logits.argmax(dim=1).item())
                else:
                    idx = int(class_idx)
                    if not 0 <= idx < n_classes:
                        raise ValueError(f"class_idx {idx} out of range 0..{n_classes - 1}")
                self.last_probs = torch.softmax(logits.detach().float(), dim=1)[0].cpu().tolist()
                logits[0, idx].backward()
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
        if self._gradients is None or self._activations is None:
            raise RuntimeError("Hooks captured nothing; is target_layer part of the forward pass?")

        self.last_class_idx = idx
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy().astype(np.float32)
        cam -= cam.min()
        peak = float(cam.max())
        self.last_cam_empty = peak <= 0.0
        if peak > 0.0:
            cam /= peak
        return cam


def overlay_heatmap(original_image, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image.convert("RGB"))
    else:
        img_np = np.array(original_image).copy()
    h, w = img_np.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_u8 = (np.clip(hm, 0.0, 1.0) * 255).astype(np.uint8)
    hm_rgb = cv2.cvtColor(cv2.applyColorMap(hm_u8, colormap), cv2.COLOR_BGR2RGB)
    out = (1 - alpha) * img_np.astype(np.float32) + alpha * hm_rgb.astype(np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def generate_gradcam_overlay(model, image_tensor, original_image, class_idx=None, alpha=0.45):
    with GradCAM(model) as g:
        heatmap = g.generate(image_tensor, class_idx=class_idx)
    return overlay_heatmap(original_image, heatmap, alpha=alpha), heatmap
