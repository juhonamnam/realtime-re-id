import torch
from .common import get_emb_similarity_score

def logistic_remap(x, threshold):
    x = x.clamp(min=1e-6, max=1 - 1e-6)
    return 1 / (1 + torch.exp(torch.log(torch.tensor(threshold / (1 - threshold))) - 
                              torch.log(x / (1 - x))))

def get_weighted_geometric_mean_similarity_score(feature1, feature2,
                         thresholds=0.5,
                         return_part_scores=False):
    ft1_v_scores, ft1_emb_vecs = feature1
    ft2_v_scores, ft2_emb_vecs = feature2

    ft_len = ft1_emb_vecs.shape[0]

    if not isinstance(thresholds, list):
        thresholds = [thresholds] * ft_len

    combined_v_scores = torch.cat((ft1_v_scores.unsqueeze(0), ft2_v_scores.unsqueeze(0)), dim=0).min(dim=0).values
    combined_v_scores_sum = combined_v_scores.sum().clamp(min=1e-6)

    total_ln_s_score = torch.tensor(0.).to(combined_v_scores.device)

    if return_part_scores:
        part_s_scores = []

    for i in range(ft_len):
        v_score = combined_v_scores[i]
        threshold = thresholds[i]

        emb_vec1 = ft1_emb_vecs[i]
        emb_vec2 = ft2_emb_vecs[i]
        s_score = get_emb_similarity_score(emb_vec1, emb_vec2)

        if s_score.isnan():
            s_score = torch.tensor(-1.).view(s_score.shape).to(s_score.device)

        if return_part_scores:
            part_s_scores.append(s_score)
        
        s_score = s_score.clamp(min=0)
        s_score = logistic_remap(s_score, threshold)

        total_ln_s_score += v_score * torch.log(s_score)

    total_s_score = torch.exp(total_ln_s_score / combined_v_scores_sum)

    if return_part_scores:
        return total_s_score, part_s_scores

    return total_s_score
