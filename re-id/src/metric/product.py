import torch

__all__ = ['get_product_similarity_score']

VALUE_AT_THRESHOLD = 0.7

def logistic_remap(x, threshold, exp=2):
    """Remaps a value using a logistic function with specific threshold behavior.

    Args:
        x (torch.Tensor): Input value(s) in range [0, 1].
        threshold (float): The value that should be mapped to VALUE_AT_THRESHOLD.
        exp (int, optional): Exponent for the logistic function. Defaults to 2.

    Returns:
        torch.Tensor: Remapped value(s).
    """
    x = x.clamp(min=1e-6, max=1 - 1e-6)
    return 1 / (1 + torch.exp(exp * torch.log(torch.tensor(threshold / (1 - threshold))) +
                              torch.log(torch.tensor((1 - VALUE_AT_THRESHOLD) / VALUE_AT_THRESHOLD)) - 
                              exp * torch.log(x / (1 - x))))

def get_product_similarity_score(v_scores, part_s_scores, part_thresholds=0.5):
    """Calculates similarity using the product principle.

    The score is the product of part similarities, where each part's influence
    is weighted by its visibility.

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

    total_s_score = torch.tensor(1.).to(v_scores.device)

    for v_score, part_s_score, part_threshold in zip(v_scores, part_s_scores, part_thresholds):
        if part_s_score.isnan():
            part_s_score = torch.tensor(-1.).view(part_s_score.shape).to(part_s_score.device)

        part_s_score = part_s_score.clamp(min=0)
        part_s_score = logistic_remap(part_s_score, part_threshold)

        total_s_score *= 1 - v_score + v_score * part_s_score

    return total_s_score
