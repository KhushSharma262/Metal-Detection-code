"""Inference: load the trained checkpoint and predict the metal class of one image.

The checkpoint is the single source of truth for architecture, class order,
input size and normalization, so inference preprocessing matches evaluation.
"""
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

from src.config import cfg
from src.model import load_checkpoint

# NOTE: training used label smoothing 0.1, so even a certain prediction tops out
# near 0.91. This threshold is only a coarse "model is unsure" flag.
CONFIDENCE_THRESHOLD = 0.60
MIN_SIDE = 32


class InvalidImageError(ValueError):
    """Raised when the input cannot be used as an image."""


class MetalPredictor:
    def __init__(self, model_path=None, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        path = Path(model_path) if model_path else Path(cfg.paths.models_dir) / cfg.paths.best_model_name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        self.model, self.ckpt = load_checkpoint(str(path), self.device)
        self.class_names = list(self.ckpt["class_names"])
        self.architecture = self.ckpt["architecture"]
        self.image_size = (int(self.ckpt["image_height"]), int(self.ckpt["image_width"]))
        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(list(self.ckpt["mean"]), list(self.ckpt["std"])),
        ])
        self.use_amp = self.device.type == "cuda"

    @staticmethod
    def _load_image(source):
        try:
            img = source if isinstance(source, Image.Image) else Image.open(source)
            img.load()
            img = img.convert("RGB")
        except FileNotFoundError as e:
            raise InvalidImageError(f"File not found: {source}") from e
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as e:
            raise InvalidImageError(f"Could not read this file as an image: {e}") from e
        if min(img.size) < MIN_SIDE:
            raise InvalidImageError(f"Image too small ({img.width}x{img.height}); minimum side is {MIN_SIDE}px.")
        return img

    def preprocess(self, source):
        """Return a (1, 3, H, W) float tensor, identical to the evaluation preprocessing."""
        return self.transform(self._load_image(source)).unsqueeze(0)

    def predict(self, source):
        x = self.preprocess(source).to(self.device)
        with torch.inference_mode():
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp):
                out = self.model(x)
        probs = torch.softmax(out.float(), dim=1)[0].cpu()
        idx = int(probs.argmax())
        conf = float(probs[idx])
        return {
            "label": self.class_names[idx],
            "index": idx,
            "confidence": round(conf, 4),
            "scores": {n: round(float(probs[i]), 4) for i, n in enumerate(self.class_names)},
            "above_threshold": conf >= CONFIDENCE_THRESHOLD,
            "class_names": self.class_names,
        }
