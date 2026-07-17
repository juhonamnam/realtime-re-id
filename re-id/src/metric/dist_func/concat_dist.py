from .abstract import DistanceFunction
import torch


class ConcatDist(DistanceFunction):
    def single(self, v_scores, part_distances):
        v_score_mask = v_scores >= self.MIN_V_SCORE
        part_distances = part_distances[v_score_mask]
        if part_distances.numel() == 0:
            return torch.tensor(-1., device=part_distances.device)
        return part_distances.norm(p=2)

    def cross_batch(self, v_scores, part_distances):
        v_score_mask = v_scores >= self.MIN_V_SCORE
        final_distances = (part_distances * v_score_mask).norm(p=2, dim=0)
        valid_mask = v_score_mask.any(dim=0)
        final_distances[~valid_mask] = -1.
        return final_distances
