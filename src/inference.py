import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.config import cfg
from src.model import build_model

CONFIDENCE_THRESHOLD = 0.60

def load_class_labels():
    with open(cfg.paths.class_labels, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

class MetalPredictor:
    def __init__(self, model_path=None, device=None):
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path     = model_path or cfg.paths.best_model
        checkpoint     = torch.load(model_path, map_location=self.device)
        num_classes    = checkpoint.get("num_classes", len(cfg.dataset.classes))
        self.model     = build_model(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        try:
            self.idx_to_class = load_class_labels()
        except FileNotFoundError:
            self.idx_to_class = {int(k): v for k, v in
                                  checkpoint.get("class_labels", {}).items()}
        self.transform = transforms.Compose([
            transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        print(f"[INFO] Model loaded | Classes: {self.idx_to_class}")

    def predict(self, image: Image.Image) -> dict:
        t = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(t), dim=1)[0]
        idx  = probs.argmax().item()
        conf = probs[idx].item()
        return {
            "label":           self.idx_to_class.get(idx, f"class_{idx}"),
            "confidence":      round(conf, 4),
            "scores":          {self.idx_to_class.get(i, f"class_{i}"): round(probs[i].item(), 4)
                                 for i in range(len(probs))},
            "above_threshold": conf >= CONFIDENCE_THRESHOLD,
        }
