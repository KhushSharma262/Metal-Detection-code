from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              precision_score, recall_score)
from src.config import cfg
from src.dataset_loader import build_dataloaders
from src.model import load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        preds  = model(images).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, save_path="logs/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)*2),
                                    max(5, len(class_names)*2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([c.capitalize() for c in class_names], fontsize=12)
    ax.set_yticklabels([c.capitalize() for c in class_names], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    thresh  = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black",
                    fontsize=11, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Confusion matrix saved -> {save_path}")


def evaluate(model_path=cfg.paths.best_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(model_path, device)
    class_names = checkpoint["class_names"]
    _, _, test_loader, _ = build_dataloaders()
    print("[Evaluate] Running inference on test set ...")
    preds, labels = collect_predictions(model, test_loader, device)
    accuracy  = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)
    f1        = f1_score(labels, preds, average="macro", zero_division=0)
    cm        = confusion_matrix(labels, preds)
    report    = classification_report(labels, preds,
                                      target_names=class_names, digits=4)
    print(f"\n{'='*55}")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'='*55}")
    print(report)
    plot_confusion_matrix(cm, class_names)
    Path("logs/evaluation_report.txt").write_text(
        f"Accuracy:{accuracy:.6f}\nPrecision:{precision:.6f}\n"
        f"Recall:{recall:.6f}\nF1:{f1:.6f}\n\n{report}")
    print("[Evaluate] Report saved -> logs/evaluation_report.txt")
    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "cm": cm}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=cfg.paths.best_model)
    args = parser.parse_args()
    evaluate(model_path=args.model)
