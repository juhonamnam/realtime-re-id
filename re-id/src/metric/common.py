import torch

def get_part_distances(emb_vecs1, emb_vecs2):
    part_distances = (emb_vecs1 - emb_vecs2).norm(p=2, dim=1)
    return part_distances

def prepare_for_distance_metric(feature1: tuple, feature2: tuple) -> tuple:
    """Prepares features from two images for distance metric calculation.

    Calculates individual part distances and combined visibility scores.

    Args:
        feature1 (tuple): (v_scores, emb_vecs)
        feature2 (tuple): (v_scores, emb_vecs)
    Returns:
        tuple: (v_scores, part_distances)
            v_scores (torch.Tensor): Minimum visibility score for each part across both images.
            part_distances (torch.Tensor): Distance for each part.
    """
    ft1_v_scores, ft1_emb_vecs = feature1
    ft2_v_scores, ft2_emb_vecs = feature2

    part_distances = get_part_distances(ft1_emb_vecs, ft2_emb_vecs)

    v_scores = torch.cat((ft1_v_scores.unsqueeze(0), ft2_v_scores.unsqueeze(0)), dim=0).amin(dim=0)

    return v_scores, part_distances
