from __future__ import annotations
import cv2
import numpy as np
import torch
from PIL import Image
from src.model import MetalClassifier


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model        = model
        self.model.eval()
        self.target_layer = target_layer or model.features[-1]
        self._activations = None
        self._gradients   = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd(module, input, output):
            self._activations = output.detach()
        def bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_full_backward_hook(bwd)

    def generate(self, image_tensor, class_idx=None, device=None):
        if device is None:
            device = next(self.model.parameters()).device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        self.model.zero_grad()
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[0, class_idx].backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        H = W   = image_tensor.shape[-1]
        cam     = torch.nn.functional.interpolate(
            cam, size=(H, W), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.astype(np.float32)


def overlay_heatmap(original_image, heatmap, alpha=0.45,
                    colormap=cv2.COLORMAP_JET):
    if isinstance(original_image, Image.Image):
        img_np = np.array(original_image.convert("RGB"))
    else:
        img_np = original_image.copy()
    h, w = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = ((1-alpha)*img_np.astype(np.float32) +
                alpha*heatmap_rgb.astype(np.float32))
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def generate_gradcam_overlay(model, image_tensor, original_image,
                              class_idx=None, alpha=0.45):
    gcam    = GradCAM(model)
    heatmap = gcam.generate(image_tensor, class_idx=class_idx)
    overlay = overlay_heatmap(original_image, heatmap, alpha=alpha)
    return overlay, heatmap
