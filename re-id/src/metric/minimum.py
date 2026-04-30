import torch

def logistic_remap(x, threshold):
    x = x.clamp(min=1e-6, max=1 - 1e-6)
    return 1 / (1 + torch.exp(torch.log(torch.tensor(threshold / (1 - threshold))) - 
                              torch.log(x / (1 - x))))

def get_minimum_similarity_score(v_scores, part_s_scores, thresholds=0.5):
    ft_len = v_scores.shape[0]

    if not isinstance(thresholds, list):
        thresholds = [thresholds] * ft_len

    # We use a high value as initial minimum
    total_s_score = torch.tensor(1.).to(v_scores.device)

    for v_score, part_s_score, threshold in zip(v_scores, part_s_scores, thresholds):
        if part_s_score.isnan():
            part_s_score = torch.tensor(-1.).view(part_s_score.shape).to(part_s_score.device)

        part_s_score = part_s_score.clamp(min=0)
        part_s_score = logistic_remap(part_s_score, threshold)

        # Consider visibility: if visibility is low, we ignore this part for the minimum
        # Following the pattern in weakest_link: 1 - v + v * s
        part_min_score = 1 - v_score + v_score * part_s_score
        total_s_score = torch.min(total_s_score, part_min_score)

    return total_s_score
