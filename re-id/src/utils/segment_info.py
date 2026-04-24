# DensePose part ids used throughout the project when merging fine-grained body
# masks into coarser segment groups.
MASK_LABELS = {
    0: "Torso",
    1: "Right Hand",
    2: "Left Hand",
    3: "Left Foot",
    4: "Right Foot",
    5: "Right Upper Leg",
    6: "Left Upper Leg",
    7: "Right Lower Leg",
    8: "Left Lower Leg",
    9: "Left Upper Arm",
    10: "Right Upper Arm",
    11: "Left Lower Arm",
    12: "Right Lower Arm",
    13: "Head",
}

def get_segment_groups(variant):
    """Return the segmentation layout used for a given model variant.

    Each group describes one output channel in the segmentation target/model:
    normalized RGBA color for visualization, the DensePose mask indices merged
    into that channel, whether the channel is synthesized as background, and a
    human-readable name.

    Variant names follow the number of non-background attention regions used by
    the model: "2a" -> 2 regions, "4a" -> 4 regions, "5a" -> 5 regions.
    """
    if variant == "2a":
        return [{"color": [0/255, 12/255, 55/255, 0.5],
                 "dp_mask_indices": [],
                 "is_background": True,
                 "name": "background"},
                # Torso and both arms are treated as one upper-body region.
                {"color": [47/255, 150/255, 224/255, 0.5],
                 "dp_mask_indices": [0, 1, 2, 9, 10, 11, 12],
                 "is_background": False,
                 "name": "upper body",
                 "default_threshold": 0.6},
                # All leg and foot parts are collapsed into one lower-body region.
                {"color": [28/255, 219/255, 169/255, 0.5],
                 "dp_mask_indices": [3, 4, 5, 6, 7, 8],
                 "is_background": False,
                 "name": "lower body",
                 "default_threshold": 0.6}]
    if variant == "4a":
        return [{"color": [0/255, 12/255, 55/255, 0.5],
                 "dp_mask_indices": [],
                 "is_background": True,
                 "name": "background"},
                # Torso is isolated so the model can attend to the central body.
                {"color": [47/255, 150/255, 224/255, 0.5],
                 "dp_mask_indices": [0],
                 "is_background": False,
                 "name": "torso",
                 "default_threshold": 0.6},
                {"color": [100/255, 255/255, 0/255, 0.5],
                 "dp_mask_indices": [1, 2, 9, 10, 11, 12],
                 "is_background": False,
                 "name": "arm",
                 "default_threshold": 0.5},
                {"color": [28/255, 219/255, 169/255, 0.5],
                 "dp_mask_indices": [5, 6],
                 "is_background": False,
                 "name": "upper leg",
                 "default_threshold": 0.5},
                {"color": [212/255, 24/255, 100/255, 0.5],
                 "dp_mask_indices": [3, 4, 7, 8],
                 "is_background": False,
                 "name": "lower leg",
                 "default_threshold": 0.5}]
    if variant == "5a":
        return [{"color": [0/255, 12/255, 55/255, 0.5],
                 "dp_mask_indices": [],
                 "is_background": True,
                 "name": "background"},
                # "5a" adds a dedicated head region on top of the "4a" split.
                {"color": [255/255, 165/255, 0/255, 0.5],
                 "dp_mask_indices": [13],
                 "is_background": False,
                 "name": "head",
                 "default_threshold": 0.5},
                {"color": [47/255, 150/255, 224/255, 0.5],
                 "dp_mask_indices": [0],
                 "is_background": False,
                 "name": "torso",
                 "default_threshold": 0.6},
                {"color": [100/255, 255/255, 0/255, 0.5],
                 "dp_mask_indices": [1, 2, 9, 10, 11, 12],
                 "is_background": False,
                 "name": "arm",
                 "default_threshold": 0.5},
                {"color": [28/255, 219/255, 169/255, 0.5],
                 "dp_mask_indices": [5, 6],
                 "is_background": False,
                 "name": "upper leg",
                 "default_threshold": 0.5},
                {"color": [212/255, 24/255, 100/255, 0.5],
                 "dp_mask_indices": [3, 4, 7, 8],
                 "is_background": False,
                 "name": "lower leg",
                 "default_threshold": 0.5},
                 ]

    raise ValueError(f"Unknown variant: {variant}")

def get_attention_groups(variant):
    """Return only foreground groups, annotated with their segmentation index.

    The re-identification model uses these entries to build one attention head
    per non-background segmentation channel while keeping the original channel
    index for mask selection.
    """
    return [{**seg, "seg_idx": idx} for idx, seg in enumerate(get_segment_groups(variant)) if not seg["is_background"]]

def get_default_thresholds(variant):
    """Return the default thresholds for each non-background segment group."""
    return [seg["default_threshold"] for seg in get_segment_groups(variant) if not seg["is_background"]]