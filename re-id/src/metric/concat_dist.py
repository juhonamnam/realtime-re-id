import torch

__all__ = ['get_concat_dist']

MIN_V_SCORE = 0.1


def get_concat_dist(v_scores, part_distances):
    v_score_mask = v_scores >= MIN_V_SCORE
    part_distances = part_distances[v_score_mask]
    if part_distances.numel() == 0:
        return torch.tensor(-1., device=part_distances.device)
    return part_distances.norm(p=2)
