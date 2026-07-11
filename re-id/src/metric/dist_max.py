import torch

__all__ = ['get_dist_max']

MIN_V_SCORE = 0.1


def get_dist_max(v_scores, part_distances):
    v_score_mask = v_scores >= MIN_V_SCORE
    part_distances = part_distances[v_score_mask]
    if part_distances.numel() == 0:
        return torch.tensor(-1., device=part_distances.device)
    return part_distances.max()
