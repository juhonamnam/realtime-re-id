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
    if "f" in variant:
        variant = variant.replace("f", "")

    if variant == "2p":
        return [{"color": [0/255, 12/255, 55/255, 0.5],
                 "dp_mask_indices": [],
                 "is_background": True,
                 "name": "background"},
                {"color": [47/255, 150/255, 224/255, 0.5],
                 "dp_mask_indices": [0, 1, 2, 9, 10, 11, 12],
                 "is_background": False,
                 "name": "upper body"},
                {"color": [28/255, 219/255, 169/255, 0.5],
                 "dp_mask_indices": [3, 4, 5, 6, 7, 8],
                 "is_background": False,
                 "name": "lower body"}]
    if variant == "3p":
        return [{"color": [0/255, 12/255, 55/255, 0.5],
                 "dp_mask_indices": [],
                 "is_background": True,
                 "name": "background"},
                {"color": [255/255, 165/255, 0/255, 0.5],
                 "dp_mask_indices": [13],
                 "is_background": False,
                 "name": "head"},
                {"color": [47/255, 150/255, 224/255, 0.5],
                 "dp_mask_indices": [0, 1, 2, 9, 10, 11, 12],
                 "is_background": False,
                 "name": "upper body"},
                {"color": [28/255, 219/255, 169/255, 0.5],
                 "dp_mask_indices": [3, 4, 5, 6, 7, 8],
                 "is_background": False,
                 "name": "lower body"}]
    if variant == "4p":
        return [{"color": [0/255, 12/255, 55/255, 0.5],
                 "dp_mask_indices": [],
                 "is_background": True,
                 "name": "background"},
                {"color": [47/255, 150/255, 224/255, 0.5],
                 "dp_mask_indices": [0],
                 "is_background": False,
                 "name": "torso"},
                {"color": [100/255, 255/255, 0/255, 0.5],
                 "dp_mask_indices": [1, 2, 9, 10, 11, 12],
                 "is_background": False,
                 "name": "arm"},
                {"color": [28/255, 219/255, 169/255, 0.5],
                 "dp_mask_indices": [5, 6],
                 "is_background": False,
                 "name": "upper leg"},
                {"color": [212/255, 24/255, 100/255, 0.5],
                 "dp_mask_indices": [3, 4, 7, 8],
                 "is_background": False,
                 "name": "lower leg"}]
    if variant == "5p":
        return [{"color": [0/255, 12/255, 55/255, 0.5],
                 "dp_mask_indices": [],
                 "is_background": True,
                 "name": "background"},
                {"color": [255/255, 165/255, 0/255, 0.5],
                 "dp_mask_indices": [13],
                 "is_background": False,
                 "name": "head"},
                {"color": [47/255, 150/255, 224/255, 0.5],
                 "dp_mask_indices": [0],
                 "is_background": False,
                 "name": "torso"},
                {"color": [100/255, 255/255, 0/255, 0.5],
                 "dp_mask_indices": [1, 2, 9, 10, 11, 12],
                 "is_background": False,
                 "name": "arm"},
                {"color": [28/255, 219/255, 169/255, 0.5],
                 "dp_mask_indices": [5, 6],
                 "is_background": False,
                 "name": "upper leg"},
                {"color": [212/255, 24/255, 100/255, 0.5],
                 "dp_mask_indices": [3, 4, 7, 8],
                 "is_background": False,
                 "name": "lower leg"}]
    if variant == "5pa":
        return [{"color": [0/255, 12/255, 55/255, 0.5],
                 "dp_mask_indices": [],
                 "is_background": True,
                 "name": "background"},
                {"color": [255/255, 165/255, 0/255, 0.5],
                 "dp_mask_indices": [13],
                 "is_background": False,
                 "name": "head"},
                {"color": [47/255, 150/255, 224/255, 0.5],
                 "dp_mask_indices": [0],
                 "is_background": False,
                 "name": "torso"},
                {"color": [100/255, 255/255, 0/255, 0.5],
                 "dp_mask_indices": [1, 2, 9, 10, 11, 12],
                 "is_background": False,
                 "name": "arm"},
                {"color": [28/255, 219/255, 169/255, 0.5],
                 "dp_mask_indices": [5, 6, 7, 8],
                 "is_background": False,
                 "name": "leg"},
                {"color": [212/255, 24/255, 100/255, 0.5],
                 "dp_mask_indices": [3, 4],
                 "is_background": False,
                 "name": "foot"}]

    raise ValueError(f"Unknown variant: {variant}")


def get_attention_groups(variant):
    segment_groups = get_segment_groups(variant)

    attention_groups = []
    for idx, seg in enumerate(segment_groups):
        if not seg["is_background"]:
            attention_groups.append(
                {**seg, "seg_idx": idx, "is_foreground": False})
    if "f" in variant:
        attention_groups.append(
            {"is_background": False, "name": "foreground", "seg_idx": -1, "is_foreground": True})
    return attention_groups
