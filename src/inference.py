from __future__ import annotations
import argparse
import time
from pathlib import Path
import torch
from PIL import Image
from src.augmentations import get_val_transforms
from src.config import cfg
from src.gradcam import GradCAM, overlay_heatmap
from src.model import load_checkpoint


class MetalPredictor:
    def __init__(self, model_path=cfg.paths.best_model, device=None):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model, checkpoint = load_checkpoint(model_path, self.device)
        self.model.eval()
        self.class_names = checkpoint["class_names"]
        self.transform   = get_val_transforms()
        print(f"[Inference] Ready | Classes: {self.class_names}")

    def preprocess(self, image):
        tensor = self.transform(image.convert("RGB"))
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image, return_gradcam=False):
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        original_pil = image.copy()
        tensor = self.preprocess(image)

        t0 = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        prob_dict  = {name: round(probs[i].item(), 4)
                      for i, name in enumerate(self.class_names)}
        predicted  = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[predicted]

        result = {
            "probabilities": prob_dict,
            "predicted"    : predicted,
            "confidence"   : confidence,
            "inference_ms" : round(inference_ms, 2),
        }

        if return_gradcam:
            pred_idx = self.class_names.index(predicted)
            gcam     = GradCAM(self.model)
            heatmap  = gcam.generate(tensor, class_idx=pred_idx)
            overlay  = overlay_heatmap(original_pil, heatmap)
            result["gradcam_image"] = overlay
            result["raw_heatmap"]   = heatmap

        return result

    def print_result(self, result):
        print("\n" + "-"*40)
        print(f"  Predicted : {result['predicted'].upper()}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Time      : {result['inference_ms']:.1f} ms")
        for cls, prob in sorted(result["probabilities"].items(),
                                key=lambda x: -x[1]):
            bar = "█" * int(prob*30) + "░" * (30 - int(prob*30))
            print(f"    {cls:<20} {bar}  {prob*100:.1f}%")
        print("-"*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   type=str, required=True)
    parser.add_argument("--model",   type=str, default=cfg.paths.best_model)
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--save",    type=str, default=None)
    args = parser.parse_args()
    predictor = MetalPredictor(model_path=args.model)
    result    = predictor.predict(args.image, return_gradcam=args.gradcam)
    predictor.print_result(result)
    if args.gradcam and "gradcam_image" in result:
        save_path = args.save or "logs/gradcam_output.png"
        result["gradcam_image"].save(save_path)
        print(f"[GradCAM] Saved -> {save_path}")
