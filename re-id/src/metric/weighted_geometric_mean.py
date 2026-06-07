import torch

__all__ = ['get_weighted_geometric_mean_similarity_score']

def logistic_remap(x, threshold):
    """Remaps a value using a logistic function such that the threshold maps to 0.5.

    Args:
         x (torch.Tensor): Input value(s) in range [0, 1].
        threshold (float): The value that should be mapped to 0.5.

    Returns:
        torch.Tensor: Remapped value(s).
    """
    x = x.clamp(min=1e-6, max=1 - 1e-6)
    return 1 / (1 + (threshold / (1 - threshold)) * ((1 - x) / x))

def get_weighted_geometric_mean_similarity_score(v_scores, part_s_scores, part_thresholds=0.5):
    """Calculates the weighted geometric mean similarity score across all body parts.

    Args:
        v_scores (torch.Tensor): Visibility scores for each part.
        part_s_scores (list[torch.Tensor]): Similarity scores for each part.
        part_thresholds (float or list[float], optional): Threshold(s) for logistic remapping. Defaults to 0.5.

    Returns:
        torch.Tensor: Combined similarity score.
    """
    ft_len = v_scores.shape[0]

    if not isinstance(part_thresholds, list):
        part_thresholds = [part_thresholds] * ft_len

    v_scores_sum = v_scores.sum().clamp(min=1e-6)

    total_ln_s_score = torch.tensor(0.).to(v_scores.device)

    for v_score, part_s_score, part_threshold in zip(v_scores, part_s_scores, part_thresholds):
        part_s_score = part_s_score.clamp(min=0)
        part_s_score = logistic_remap(part_s_score, part_threshold)

        total_ln_s_score += v_score * torch.log(part_s_score)

    total_s_score = torch.exp(total_ln_s_score / v_scores_sum)

    return total_s_score
