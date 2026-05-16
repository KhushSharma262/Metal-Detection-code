from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.gradcam import GradCAM


METAL_TEXTURE_TRAITS = {
    "aluminum"        : ["uniform fine grain", "smooth machining marks", "low reflectivity variation", "subtle parallel striations"],
    "brass"           : ["warm tone grain", "dense cross-hatch marks", "high specular reflection", "tight crystalline texture"],
    "Copper"          : ["reddish grain clusters", "oxidation micro-patches", "soft directional scratches", "medium reflectivity"],
    "Iron"            : ["coarse dark grain", "pitting micro-defects", "rough surface texture", "low specular response"],
    "Lead"            : ["very soft grain", "wide dull surface", "low contrast texture", "smear-like tool marks"],
    "Mild Steel"      : ["medium grain density", "rolling mill lines", "moderate reflectivity", "fine surface waviness"],
    "Stainless Steel" : ["tight fine grain", "bright specular highlight", "brushed directional lines", "high contrast texture"],
    "Tool Steel EN 31": ["ultra-fine grain", "hardened surface marks", "high hardness texture", "very tight machining lines"],
    "Zinc"            : ["spangle crystal pattern", "large grain boundaries", "dull matte finish", "irregular surface topology"],
}


def get_texture_insights(class_name: str, confidence: float) -> dict:
    traits = METAL_TEXTURE_TRAITS.get(class_name, ["surface texture pattern"])
    confidence_pct = confidence * 100

    if confidence_pct >= 90:
        strength = "very strong"
    elif confidence_pct >= 75:
        strength = "strong"
    elif confidence_pct >= 60:
        strength = "moderate"
    else:
        strength = "weak"

    primary_trait   = traits[0]
    secondary_trait = traits[1] if len(traits) > 1 else traits[0]

    insight = (
        f"The model detected {strength} evidence of **{class_name}** "
        f"with {confidence_pct:.1f}% confidence. "
        f"The dominant texture signature is '{primary_trait}', "
        f"supported by '{secondary_trait}' patterns visible in the surface micro-structure."
    )

    return {
        "insight"        : insight,
        "traits"         : traits,
        "confidence_pct" : confidence_pct,
        "strength"       : strength,
        "metal"          : class_name,
    }


def compute_feature_vector(model, image_tensor: torch.Tensor, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        features = model.get_features(image_tensor)
        return features.cpu().numpy().flatten()


def compute_similarity_matrix(
    model,
    class_names: list[str],
    sample_tensors: dict[str, torch.Tensor],
    device,
) -> np.ndarray:
    vectors = {}
    for cls, tensor in sample_tensors.items():
        vectors[cls] = compute_feature_vector(model, tensor, device)

    n   = len(class_names)
    mat = np.zeros((n, n))

    for i, c1 in enumerate(class_names):
        for j, c2 in enumerate(class_names):
            if c1 in vectors and c2 in vectors:
                v1 = vectors[c1]
                v2 = vectors[c2]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                mat[i, j] = float(sim)
            else:
                mat[i, j] = 1.0 if i == j else 0.0

    return mat


def get_top_activation_regions(heatmap: np.ndarray, top_k: int = 3) -> list[dict]:
    h, w  = heatmap.shape
    regions = []
    temp  = heatmap.copy()

    region_names = ["top-left", "top-center", "top-right",
                    "mid-left", "center",     "mid-right",
                    "bot-left", "bot-center", "bot-right"]

    grid_h = h // 3
    grid_w = w // 3

    scores = []
    for ri in range(3):
        for ci in range(3):
            patch = temp[ri*grid_h:(ri+1)*grid_h, ci*grid_w:(ci+1)*grid_w]
            scores.append((patch.mean(), region_names[ri*3+ci]))

    scores.sort(key=lambda x: -x[0])
    for score, name in scores[:top_k]:
        regions.append({"region": name, "activation": float(score)})

    return regions
