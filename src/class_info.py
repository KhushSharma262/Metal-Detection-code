"""Display metadata for the nine metal classes.

The model checkpoint is the source of truth for class identity and order
(MetalPredictor.class_names). This module only stores how each class is shown
in the UI. validate_class_meta() fails loudly if the two ever disagree.
"""

CLASS_META = {
    "Aluminium": {
        "color": "#3B82F6",
        "description": "Light, silvery metal that forms a protective oxide layer. "
                       "Widely used in packaging, construction and transport.",
    },
    "Brass": {
        "color": "#EAB308",
        "description": "Yellowish copper-zinc alloy. Common in fittings, "
                       "hardware and musical instruments.",
    },
    "Copper": {
        "color": "#EA580C",
        "description": "Reddish metal with very high electrical and thermal "
                       "conductivity. Used in wiring and plumbing.",
    },
    "Iron": {
        "color": "#64748B",
        "description": "Grey metal that is the base of steels. Rusts readily "
                       "in moist air.",
    },
    "Lead": {
        "color": "#8B5CF6",
        "description": "Dense, soft, bluish-grey metal. It is toxic, so handle "
                       "with care.",
    },
    "Mild Steel": {
        "color": "#92400E",
        "description": "Low-carbon steel that is easy to machine and weld. "
                       "Very common in general fabrication.",
    },
    "Stainless Steel": {
        "color": "#0891B2",
        "description": "Iron-based alloy designed to resist corrosion. Used in "
                       "kitchenware, medical and food equipment.",
    },
    "Tool Steel EN 31": {
        "color": "#DC2626",
        "description": "Hardenable alloy steel valued for wear resistance. "
                       "Used for parts that see heavy contact and friction.",
    },
    "Zinc": {
        "color": "#16A34A",
        "description": "Bluish-white metal used for galvanizing steel and "
                       "for die casting.",
    },
}

DEFAULT_COLOR = "#888888"


def validate_class_meta(class_names):
    """Raise ValueError unless CLASS_META keys exactly match the checkpoint's class names."""
    names = set(class_names)
    meta = set(CLASS_META)
    missing = sorted(names - meta)
    extra = sorted(meta - names)
    if missing or extra:
        raise ValueError(
            "UI metadata does not match the model checkpoint. "
            f"Classes without metadata: {missing}. Metadata without a class: {extra}."
        )


def get_meta(label):
    """Return metadata for a label (falls back to a neutral entry; validate first)."""
    return CLASS_META.get(label, {"color": DEFAULT_COLOR, "description": ""})
